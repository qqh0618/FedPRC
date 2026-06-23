"""
2025/11/6 10:51
while True:
    leanring
本文件由my_ywj首次创建编写
"""
import torch
from model import clip
from model.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from model.clip.clip import build_model

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# CLIP加载及修改结构

_tokenizer = _Tokenizer()


class WJCLIP(nn.Module):
    def __init__(self, args, classnames=None, code_len=512, num_classes=7):
        super().__init__()
        if args.dataset == 'pacs':
            classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        elif args.dataset == 'caltech10':
            classes = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']
        classnames = []
        for target in classes:
            others = [c for c in classes if c != target]
            neg_desc = ", ".join(others)
            prompt = f"a photo of a {target}, not including {neg_desc}"
            classnames.append(prompt)
# → "a photo of a dog, not including cat, car, person, ..."
            # classnames = [
            #                 "a photo of dog",
            #                 "a photo of elephant",
            #                 "a photo of giraffe",
            #                 'a photo of guitar',
            #                 'a photo of horse',
            #                 'a photo of house',
            #                 'a photo of person',
            #                 ]
            
            
        design_details = {"trainer": args.alg,
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = torch.jit.load(r"pretrain/RN50.pt", map_location='cpu').eval()
        self.clip_model = build_model(model.state_dict(), design_details)
        self.prompt_learner = PromptLearner(args, classnames, self.clip_model)  # promptfl fedotp共用
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.tokenizer_txt = clip.tokenize(classnames).to('cuda')

        self.txt_proj = nn.Linear(1024, code_len).half()
        self.img_proj = nn.Linear(1024, code_len).half()


        # --------------------------------------
        # FedOTP
        self.N = args.glp_otp_n
        self.use_uniform = args.use_uniform
        self.eps = args.eps
        self.thresh = args.thresh
        self.OT = args.OT
        self.top_percent = args.top_percent
        self.max_iter = args.max_iter
        self.n_cls = len(classnames)


    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = self.thresh
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T

    def entropic_COT_fast(self, a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False):
        """
        modify from ot.partial.entropic_partial_wasserstein in torch version

        """
        dx = torch.ones_like(a)
        dy = torch.ones_like(b)

        log_e = {'err': []}
        stopThr = self.thresh

        # K = torch.exp(M / (-reg))
        K = M

        Kp = torch.matmul(torch.diag_embed(1 / a, dim1=1), K)
        Kq = torch.matmul(torch.diag_embed(1 / b, dim1=1), K.permute(0, 2, 1))

        err, cpt = 1, 0
        u = dx
        v = dy
        while (cpt < numItermax):

            v0 = v
            temp = torch.div(dx, torch.matmul(Kp, v.unsqueeze(-1)).squeeze(-1))
            u = torch.minimum(temp, dx)
            v = torch.div(dy, torch.matmul(Kq, u.unsqueeze(-1)).squeeze(-1))

            cpt = cpt + 1
            err = (v - v0).abs().mean()
            if err.item() < stopThr:
                break
        Kprev = torch.matmul(torch.diag_embed(u, dim1=1), K)
        Kprev = torch.matmul(Kprev, torch.diag_embed(v, dim1=1))
        if log:
            return Kprev, log_e
        else:
            return Kprev

    def forward_otp(self, image):
        b = image.shape[0]
        image_features = self.image_encoder(image.type(self.dtype))
        # image_features = self.img_proj(image_features)
        image_feature_pool = image_features[0]
        image_features = image_features[1:]
        M = image_features.shape[0]
        self.d = image_features.shape[-1]

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts


        text_features = self.text_encoder(prompts, tokenized_prompts)
        # text_features = self.txt_proj(text_features)
        text_features = text_features.contiguous().view(self.N, self.n_cls, self.d)
        text_feature_pool = text_features.mean(dim=0)


        image_features = F.normalize(image_features, dim=2)
        image_feature_pool = F.normalize(image_feature_pool, dim=1)
        text_features = F.normalize(text_features, dim=2)
        text_feature_pool = F.normalize(text_feature_pool, dim=1)

        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()
        sim = sim.view(M, self.N, b * self.n_cls)
        sim = sim.permute(2, 0, 1)
        wdist = 1.0 - sim

        xx = torch.zeros(b * self.n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        # if self.OT == 'Sinkhorn':
        # yy = torch.zeros(b * self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)
        # elif self.OT == 'COT':
        top_percent = min(torch.sum(xx).item(), self.top_percent)
        yy = torch.zeros(b * self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(
            1. / self.N) * top_percent

        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            # if self.OT == 'Sinkhorn':
            # T = self.Sinkhorn(KK, xx, yy)
            # elif self.OT == 'COT':
            T = self.entropic_COT_fast(xx, yy, KK, 0.01, numItermax=self.max_iter)
        if torch.isnan(T).any():
            return None

        sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b, self.n_cls)

        logit_scale = self.logit_scale.exp()*sim_op
        logits = logit_scale * (image_feature_pool @ text_feature_pool.t())

        return logits, image_features, text_features

    def forward(self, image):
        # PromptFL: Let Federated Participants Cooperatively Learn Prompts Instead of Models
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits,image_features,text_features

    def forward_clip(self, image):
        # Federated CLIP
        image_features = self.clip_model.encode_image(image)
        txt_features = self.clip_model.encode_text(self.tokenizer_txt)

        # image_features = self.img_proj(image_features)
        # txt_features = self.txt_proj(txt_features)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.t())
        return logits,image_features,text_features



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.n_ctx  # number of context words
        ctx_init = cfg.ctx_init  # initialization words for context vectors
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT_SIZE
        self.N = cfg.glp_otp_n  # number of prompts  --> FedOTP
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.cfg = cfg
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                if cfg.alg == "FedOTP":
                    ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
                else:
                    ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        if cfg.alg == "FedOTP":
            tokenized_prompts = tokenized_prompts.repeat(self.N, 1)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        if self.ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)


        if self.cfg.alg == "FedOTP":
            ctx = ctx.permute(1, 0, 2, 3)
            ctx = ctx.contiguous().view(self.N * self.n_cls, self.n_ctx, ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]

                if self.cfg.alg == "FedOTP":
                    prompt = torch.cat(
                        [
                            prefix_i,  # (1, 1, dim)
                            ctx_i_half1,  # (1, n_ctx//2, dim)
                            class_i,  # (1, name_len, dim)
                            ctx_i_half2,  # (1, n_ctx//2, dim)
                            suffix_i,  # (1, *, dim)
                        ],
                        dim=1,
                    )
                else:

                    prompt = torch.cat(
                        [
                            prefix_i,  # (1, 1, dim)
                            ctx_i_half1,  # (1, n_ctx//2, dim)
                            class_i,  # (1, name_len, dim)
                            ctx_i_half2,  # (1, n_ctx//2, dim)
                            suffix_i,  # (1, *, dim)
                        ],
                        dim=1,
                    )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts





