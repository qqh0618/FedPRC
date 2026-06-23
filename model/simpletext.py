import torch
import torch.nn as nn
import clip
import numpy as np
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

seg_labels = ["a photo of a dog",
                      "a photo of a elephant",
                      "a photo of a giraffe",
                      "a photo of a guitar",
                       "a photo of a horse",
                       "a photo of a house",
                       "a photo of a person",
                       ]

class PromptLearner(nn.Module):
    def __init__(self, classnames):
        super().__init__()
        n_cls = len(classnames)
        # n_ctx = cfg.TRAINER.PROMPTFL.N_CTX
        n_ctx = 16  # number of text encoder of text prompts
        # ctx_init = cfg.TRAINER.PROMPTFL.CTX_INIT
        ctx_init = False  # is using the ctx init, set True for CLIP
        dtype = torch.float32
        ctx_dim = 512
        print("Initializing a generic context")
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
        # with torch.no_grad():
        #     embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        embedding = nn.Embedding(tokenized_prompts.shape[1], ctx_dim)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = 'end'

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

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



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class CustomTextClassifier(nn.Module):
    def __init__(self, tokenizer, num_classes, vocab_size=49408, context_length=77, transformer_width=512, hidden_dim=512, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.prompt_learner = PromptLearner(num_classes)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, hidden_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self):

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts

        x = self.token_embedding(prompts).type(torch.float32)  # [batch_size, n_ctx, d_model]

        # ==================================
        # encoder部分
        x = x + self.positional_embedding.type(torch.float32)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(torch.float32)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        # ==================================
        return x

text = clip.tokenize(["a photo of a dog",
                      "a photo of a elephant",
                      "a photo of a giraffe",
                      "a photo of a guitar",
                       "a photo of a horse",
                       "a photo of a house",
                       "a photo of a person",
                       ])


# model = CustomTextClassifier(text, num_classes=7)
# y = model(text)
# print(y)

# embed_dim = state_dict["text_projection"].shape[1]  512
# context_length = state_dict["positional_embedding"].shape[0]  77
# vocab_size = state_dict["token_embedding.weight"].shape[0]   49408