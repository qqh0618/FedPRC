import copy

import torch

from core.Client.tta.ClientFedATP import ClientFedATP
from core.Server.ServerBase import Server
from core.Client.ClientFedAvg import ClientFedAvg
from tqdm import tqdm
import numpy as np
from utils import average_weights
from mem_utils import MemReporter
import time
import torch.nn as nn
from copy import deepcopy
from typing import List, Optional, Tuple
import torch.nn.functional as F


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    H(p) = -∑ p(y) log p(y)
    Minimizing entropy pushes the model toward confident predictions.
    This is the standard unsupervised loss for test-time adaptation.
    """
    p = F.softmax(logits, dim=1)
    return -(p * torch.log(p + 1e-12)).sum(dim=1).mean()


# ═════════════════════════════════════════════════════════════════════════════
# Utility: get trainable params + classify indices
# ═════════════════════════════════════════════════════════════════════════════

def _get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """Return trainable parameters in deterministic order (no side-effects)."""
    return [p for _, p in model.named_parameters() if p.requires_grad]


def _get_param_stat_indices(model: nn.Module) -> Tuple[List[int], List[int]]:
    """
    Classify trainable parameter indices into:
      - param_indices: weights/biases (everything except BN running stats)
      - stat_indices:  BatchNorm running_mean / running_var
    """
    param_idx, stat_idx = [], []
    for i, (name, p) in enumerate(model.named_parameters()):
        if not p.requires_grad:
            continue
        if 'running' in name:
            stat_idx.append(i)
        else:
            param_idx.append(i)
    return param_idx, stat_idx


class MyBatchNorm2d(nn.Module):

    def __init__(self, bn):
        super(MyBatchNorm2d, self).__init__()

        self.num_features = bn.num_features
        self.eps = bn.eps

        self.num_batches_tracked = bn.num_batches_tracked

        self.running_mean = nn.parameter.Parameter(bn.running_mean.detach().clone(), True)
        self.running_var = nn.parameter.Parameter(bn.running_var.detach().clone(), True)

        self.weight = nn.parameter.Parameter(bn.weight.detach().clone())
        self.bias = nn.parameter.Parameter(bn.bias.detach().clone())

        self.snapshot_mean = None
        self.snapshot_var = None

        self.training = True

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False

    def forward(self, x):
        # save feature statistics for backward
        with torch.no_grad():
            x_t = x.data.permute((1, 0, 2, 3)).reshape((self.num_features, -1)).detach().clone()
            self.snapshot_mean = x_t.mean(dim=1)
            self.snapshot_var = x_t.var(dim=1)
            self.num_batches_tracked += 1

        if not self.training:
            x = (x - self.running_mean.view((-1, 1, 1))) / torch.sqrt(self.running_var.view((-1, 1, 1)) + self.eps)

        else:
            var_forward = x_t.var(dim=1, unbiased=False)
            x = (x - self.snapshot_mean.view((-1, 1, 1))) / torch.sqrt(var_forward.view((-1, 1, 1)) + self.eps)

        x = x * self.weight.view((-1, 1, 1)) + self.bias.view((-1, 1, 1))

        return x

    def set_running_stat_grads(self):
        with torch.no_grad():
            self.running_mean.grad = self.running_mean.data - self.snapshot_mean
            self.running_var.grad = self.running_var.data - self.snapshot_var
            # self.running_var.grad = torch.zeros_like(self.running_var.grad)

    def clip_running_var(self):
        with torch.no_grad():
            self.running_var.clamp_(min=0)


class ATPLearner:
    """
    Learns per-parameter adaptation rates α on source-client data.

    α[i] controls how far parameter θ[i] moves along its unsupervised gradient.
    The bilevel objective:
        min_α  𝔼_{(x,y)~source} L_ce(θ - α⊙∇L_ent(θ; x), y)

    Intuition: if moving θ[i] along the unsupervised gradient direction also
    reduces supervised loss, α[i] should be large. Otherwise, small.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self._params = _get_trainable_params(self.model)
        self._num_params = len(self._params)

        # Separate indices for weight/bias params vs BN running stats
        self._param_indices, self._stat_indices = _get_param_stat_indices(self.model)

        print(f'[ATP] Model has {self._num_params} trainable parameters')
        print(f'      Params (weights/biases): {len(self._param_indices)}')
        print(f'      BN running stats:        {len(self._stat_indices)}')

    @property
    def num_adaptation_rates(self) -> int:
        return self._num_params

    def learn(
        self,
        source_loaders: List[torch.utils.data.DataLoader],
        num_rounds: int = 200,
        cohort_size: float = 0.25,
        meta_lr: float = 0.1,
        grad_norm: str = 'sqrt_numel',
        verbose: bool = True,
        eval_loaders: Optional[List[torch.utils.data.DataLoader]] = None,
        eval_mode: str = 'batch',
    ) -> torch.Tensor:
        """
        Learn adaptation rates.

        Args:
            source_loaders: list of DataLoaders, one per source client
                            (use their *validation* split, not training split)
            num_rounds: number of meta-learning communication rounds
            cohort_size: fraction of clients sampled each round (0 < c ≤ 1)
            meta_lr: learning rate for the adaptation rates themselves
            grad_norm: 'none' | 'numel' | 'sqrt_numel' (default, from paper)
            verbose: print progress
            eval_loaders: optional target-client loaders for validation
            eval_mode: 'batch' (ATP-batch) or 'online_avg' (ATP-online)

        Returns:
            adapt_lrs: tensor of shape (num_params,) — learned per-parameter rates
        """
        # Re-fetch params (model may have changed since __init__)
        self._params = _get_trainable_params(self.model)

        # Init adaptation rates to zero
        adapt_lrs = torch.zeros(self._num_params, device=self.device)

        # Per-parameter normalizer (used by some grad_norm modes)
        numels = torch.tensor(
            [p.numel() for p in self._params],
            device=self.device, dtype=torch.float32,
        )

        for rnd in range(1, num_rounds + 1):
            # ---- Sample a cohort of source clients ----
            num_sample = max(1, round(len(source_loaders) * cohort_size))
            indices = torch.randperm(len(source_loaders))[:num_sample].tolist()

            global_state = deepcopy(self.model.state_dict())
            global_adapt_lrs = adapt_lrs.clone()
            accum_adapt_lrs = torch.zeros_like(adapt_lrs)

            for idx in indices:
                # Reset model + adapt_lrs to global before each client
                self.model.load_state_dict(global_state, strict=False)
                adapt_lrs.copy_(global_adapt_lrs)

                # Bilevel update on this client's data
                self._client_update(
                    dataloader=source_loaders[idx],
                    adapt_lrs=adapt_lrs,
                    meta_lr=meta_lr,
                    grad_norm=grad_norm,
                    numels=numels,
                )
                accum_adapt_lrs += adapt_lrs

            # ---- Aggregate: average across sampled clients (FedAvg-style) ----
            adapt_lrs = accum_adapt_lrs / num_sample
            self.model.load_state_dict(global_state, strict=False)

            if verbose:
                self._log_progress(rnd, num_rounds, adapt_lrs,
                                   eval_loaders, eval_mode)

        return adapt_lrs

    def _client_update(
        self,
        dataloader: torch.utils.data.DataLoader,
        adapt_lrs: torch.Tensor,
        meta_lr: float,
        grad_norm: str,
        numels: torch.Tensor,
    ):
        """
        One client's bilevel update of adapt_lrs.

        For each batch (x, y):
          1. θ' = θ - α ⊙ ∇L_ent(θ; x)          # unsupervised adaptation
          2. ∇_α L_ce(θ'; y) via chain rule     # meta-gradient
          3. α ← α + meta_lr * ∇_α
        """
        state = deepcopy(self.model.state_dict())

        for *X, Y in dataloader:
            self.model.load_state_dict(state)

            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            # ---- Step 1: unsupervised adaptation ----
            logits = self.model(*X)
            loss_ent = entropy_loss(logits)
            loss_ent.backward()

            # let BN running stats receive gradient toward batch statistics
            if hasattr(self.model, 'set_running_stat_grads'):
                self.model.set_running_stat_grads()

            unspv_grad = [p.grad.clone() for p in self._params]

            # θ' = θ - α ⊙ g_unsup
            with torch.no_grad():
                for i, (p, g) in enumerate(zip(self._params, unspv_grad)):
                    p -= adapt_lrs[i] * g

            self.model.zero_grad()

            # prevent negative variance → NaN
            if hasattr(self.model, 'clip_bn_running_vars'):
                self.model.clip_bn_running_vars()

            # ---- Step 2: supervised loss on adapted model ----
            self.model.eval()
            logits_adapted = self.model(*X)
            loss_spv = F.cross_entropy(logits_adapted, Y)

            spv_grad = torch.autograd.grad(
                loss_spv, list(self._params), create_graph=False,
            )

            # ---- Step 3: update α via chain rule ----
            # ∂/∂α[i] L_ce(θ - α⊙g) = - spv_grad[i] · unspv_grad[i]
            # Gradient *ascent* direction for α:  + spv_grad[i] · unspv_grad[i]
            with torch.no_grad():
                g_meta = self._compute_meta_gradient(
                    spv_grad, unspv_grad, grad_norm, numels,
                )
                adapt_lrs += meta_lr * g_meta

    def _compute_meta_gradient(
        self,
        spv_grad: List[torch.Tensor],
        unspv_grad: List[torch.Tensor],
        grad_norm: str,
        numels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute g[i] = (spv_grad[i] · unspv_grad[i]) / norm[i]"""
        g = torch.zeros(len(spv_grad), device=self.device)

        for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
            dot = (g1 * g2).sum()

            if grad_norm == 'none':
                g[i] = dot
            elif grad_norm == 'numel':
                g[i] = dot / g1.numel()
            elif grad_norm == 'sqrt_numel':
                g[i] = dot / (g1.numel() ** 0.5)
            else:
                raise ValueError(f'Unknown grad_norm: {grad_norm}')

        return g

    def _log_progress(self, rnd, num_rounds, adapt_lrs, eval_loaders, eval_mode):
        """Print round-level statistics."""
        if rnd == 1 or rnd % 10 == 0:
            stats = (
                f'Round {rnd:4d}/{num_rounds} | '
                f'α mean={adapt_lrs.mean().item():.6f}  '
                f'max={adapt_lrs.max().item():.6f}  '
                f'min={adapt_lrs.min().item():.6f}'
            )
            if eval_loaders is not None:
                adapter = ATPAdapter(self.model, adapt_lrs, device=str(self.device))
                accs = []
                for loader in eval_loaders:
                    acc, _ = adapter.evaluate(loader, mode=eval_mode)
                    accs.append(acc)
                stats += f' | target acc={sum(accs)/len(accs):.2f}%'
            print(stats)


# ═════════════════════════════════════════════════════════════════════════════
# ATPAdapter: test-time adaptation with learned rates
# ═════════════════════════════════════════════════════════════════════════════

class ATPAdapter:
    """
    Apply ATP test-time adaptation with pre-learned per-parameter rates.

    Two modes:
      - 'batch' (ATP-batch):   reset to global model before EACH batch
      - 'online_avg' (ATP-online): maintain exponential moving average of
        model state across batches, with decaying adaptation rate
    """

    def __init__(
        self,
        model: nn.Module,
        adapt_lrs: torch.Tensor,
        device: str = 'cuda',
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.adapt_lrs = adapt_lrs.to(self.device)
        self._params = _get_trainable_params(self.model)

        assert len(self._params) == len(self.adapt_lrs), (
            f'adapt_lrs has {len(self.adapt_lrs)} entries '
            f'but model has {len(self._params)} trainable parameters. '
            f'Did you call replace_bn_with_grad() on the model?'
        )

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        mode: str = 'batch',
        progress: bool = False,
    ) -> Tuple[float, float]:
        """
        Evaluate on one target client.

        Args:
            dataloader: DataLoader for a single target client
            mode: 'batch' for ATP-batch, 'online_avg' for ATP-online
            progress: show tqdm progress bar

        Returns:
            (accuracy_percent, avg_loss)
        """
        if mode == 'batch':
            return self._eval_batch(dataloader, progress)
        elif mode == 'online_avg':
            return self._eval_online_avg(dataloader, progress)
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Use 'batch' or 'online_avg'."
            )

    # ------------------------------------------------------------------
    # ATP-batch: independent adaptation per batch
    # ------------------------------------------------------------------

    def _eval_batch(self, dataloader, progress: bool) -> Tuple[float, float]:
        global_state = deepcopy(self.model.state_dict())

        total, total_loss, total_correct = 0, 0.0, 0
        iterator = self._maybe_tqdm(dataloader, progress, desc='ATP-batch')

        for *X, Y in iterator:
            self.model.load_state_dict(global_state)

            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            # Unsupervised adaptation step
            self._adapt_one_step(X, Y)

            # Supervised evaluation
            self.model.eval()
            with torch.no_grad():
                logits = self.model(*X)
                loss = F.cross_entropy(logits, Y)

            n = len(X[0])
            total += n
            total_loss += loss.item() * n
            total_correct += (logits.argmax(1) == Y).sum().item()

        return total_correct / total * 100, total_loss / total

    # ------------------------------------------------------------------
    # ATP-online: EMA of model state across batches
    # ------------------------------------------------------------------

    def _eval_online_avg(self, dataloader, progress: bool) -> Tuple[float, float]:
        global_state = deepcopy(self.model.state_dict())

        total, total_loss, total_correct = 0, 0.0, 0
        iterator = self._maybe_tqdm(dataloader, progress, desc='ATP-online')

        for i, (*X, Y) in enumerate(iterator):
            if i == 0:
                acc_state = deepcopy(global_state)
            else:
                acc_state = deepcopy(self.model.state_dict())
                self.model.load_state_dict(global_state)

            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            # Decayed adaptation rate + partial state reset
            current_lrs = self.adapt_lrs * 0.5
            self.model.load_state_dict(
                self._wavg_state(global_state, self.model.state_dict(), 0.5)
            )

            # Unsupervised adaptation step
            self._adapt_one_step(X, Y, override_lrs=current_lrs)

            # EMA update after adaptation
            state_now = self.model.state_dict()
            state_new = self._wavg_state(acc_state, state_now, i / (i + 1))
            self.model.load_state_dict(state_new)

            # Evaluation
            self.model.eval()
            with torch.no_grad():
                logits = self.model(*X)
                loss = F.cross_entropy(logits, Y)

            n = len(X[0])
            total += n
            total_loss += loss.item() * n
            total_correct += (logits.argmax(1) == Y).sum().item()

        return total_correct / total * 100, total_loss / total

    # ------------------------------------------------------------------
    # Core: one step of unsupervised adaptation
    # ------------------------------------------------------------------

    def _adapt_one_step(
        self,
        X: List[torch.Tensor],
        Y: torch.Tensor,
        override_lrs: Optional[torch.Tensor] = None,
    ):
        """
        θ ← θ - α ⊙ ∇L_ent(θ; x)

        Uses entropy minimization. BN running stats get their gradients
        set manually via set_running_stat_grads() (moving toward batch stats).
        """
        lrs = override_lrs if override_lrs is not None else self.adapt_lrs

        self.model.eval()
        logits = self.model(*X)
        loss_ent = entropy_loss(logits)
        loss_ent.backward()

        if hasattr(self.model, 'set_running_stat_grads'):
            self.model.set_running_stat_grads()

        unspv_grad = [p.grad.clone() for p in self._params]

        with torch.no_grad():
            for i, (p, g) in enumerate(zip(self._params, unspv_grad)):
                p -= lrs[i] * g

        self.model.zero_grad()

        if hasattr(self.model, 'clip_bn_running_vars'):
            self.model.clip_bn_running_vars()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _wavg_state(s1: dict, s2: dict, lam: float) -> dict:
        """Weighted average: lam * s1 + (1 - lam) * s2."""
        out = deepcopy(s1)
        for k in s1:
            out[k] = lam * s1[k] + (1 - lam) * s2[k]
        return out

    @staticmethod
    def _maybe_tqdm(iterable, enable: bool, **kwargs):
        if enable:
            from tqdm import tqdm
            return tqdm(iterable, **kwargs)
        return iterable


def replace_bn_with_grad(model: nn.Module, verbose: bool = False) -> nn.Module:
    """
    Recursively replace all nn.BatchNorm2d layers in `model` with MyBatchNorm2d.

    Handles:
      - Plain attributes:      model.bn1 = MyBatchNorm2d(model.bn1)
      - Sequential children:   model.layer1[0].bn1 = MyBatchNorm2d(...)
      - Downsample shortcuts:  model.layer2[0].downsample[1] = MyBatchNorm2d(...)

    Args:
        model: any nn.Module containing nn.BatchNorm2d layers
        verbose: print names of replaced layers

    Returns:
        model (modified in-place)
    """
    _replace_bn_recursive(model, '', verbose)
    return model


def _replace_bn_recursive(module: nn.Module, prefix: str, verbose: bool):
    """Walk the module tree and replace BatchNorm2d children."""
    for name, child in list(module.named_children()):
        full_name = f'{prefix}.{name}' if prefix else name

        if isinstance(child, nn.BatchNorm2d):
            if verbose:
                print(f'  [replace_bn] {full_name}: nn.BatchNorm2d → MyBatchNorm2d')
            setattr(module, name, MyBatchNorm2d(child))

        elif isinstance(child, (nn.Sequential, nn.ModuleList)):
            # Handle indexed containers: layer[i].bn, downsample[i], etc.
            for idx, sub_child in enumerate(child):
                if isinstance(sub_child, nn.BatchNorm2d):
                    if verbose:
                        print(f'  [replace_bn] {full_name}[{idx}]: nn.BatchNorm2d → MyBatchNorm2d')
                    child[idx] = MyBatchNorm2d(sub_child)
                else:
                    _replace_bn_recursive(sub_child, f'{full_name}[{idx}]', verbose)
        else:
            _replace_bn_recursive(child, full_name, verbose)


# ---------------------------------------------------------------------------
# Convenience: attach helper methods to any model
# ---------------------------------------------------------------------------

def attach_bn_helpers(model: nn.Module):
    """
    Attach set_running_stat_grads() and clip_bn_running_vars() to `model`
    so you don't need to keep references to the ATP wrapper.

    After calling this, you can do:
        model.set_running_stat_grads()
        model.clip_bn_running_vars()
    """

    def _set_running_stat_grads():
        for m in model.modules():
            if isinstance(m, MyBatchNorm2d):
                m.set_running_stat_grads()

    def _clip_bn_running_vars():
        for m in model.modules():
            if isinstance(m, MyBatchNorm2d):
                m.clip_running_var()

    model.set_running_stat_grads = _set_running_stat_grads
    model.clip_bn_running_vars = _clip_bn_running_vars
    return model



def replace_bn_with_custom(model):
    for name, module in model.named_children():
        # 如果当前子模块是标准的 BatchNorm2d
        if isinstance(module, nn.BatchNorm2d):
            # 直接使用原有模块实例化你的自定义 BN
            custom_bn = MyBatchNorm2d(module)

            # 将原模型中的该模块替换为自定义模块
            setattr(model, name, custom_bn)
        else:
            # 递归处理嵌套的子模块（如 Sequential, ResNet Block 等）
            replace_bn_with_custom(module)

class ServerFedATP(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map)
        # --------------------------------
        # 替换模型的BN层
        self.global_model.load_state_dict(torch.load(self.args.model_path,map_location='cpu'))
        attach_bn_helpers(self.global_model)
        learner = ATPLearner(self.global_model, device=device)
        # --------------------------------

    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedATP(self.args, copy.deepcopy(self.global_model),self.net_idx_dataidx_map[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))
            
            
    def train(self):
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []

        for epoch in tqdm(range(self.args.comm_round)):
            test_accuracy = 0
            local_weights, local_losses = [], []
            self.logger.info(f'Global Training Round: {epoch+1}')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)
            global_weights = self.global_model.state_dict()
            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)
                w, loss = self.LocalModels[idx].update_weights(global_round=epoch)
                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                acc = self.LocalModels[idx].test_accuracy()
                test_accuracy += acc

            # update global weights
            global_weights = self.average_weights(local_weights, idxs_users)
            self.global_model.load_state_dict(global_weights)

            # print loss
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            cur_g_acc = self.global_test_accuracy()
            if cur_g_acc > self.global_best_acc:
                self.global_best_acc = cur_g_acc
                self.Save_CheckPoint(self.args.logdir + '/best_model.pth')
            if test_accuracy / len(idxs_users)  > self.global_best_personal_acc:
                self.global_best_personal_acc = test_accuracy / len(idxs_users)
            self.logger.info(f'Global Training Loss: {loss_avg}')
            self.logger.info(
                f'Personal_Accuracy: {test_accuracy / len(idxs_users)} || Best_Personal_Accuracy: {self.global_best_personal_acc}')
            self.logger.info(f'Global_Accuracy: {cur_g_acc} || Best_Accuracy: {self.global_best_acc}')
            self.csv_log(
                {'DataName': f"{self.args.dataset}_{self.args.leave_domain}", 'Round': epoch + 1, 'Loss': loss_avg,
                 'Personal_Acc': test_accuracy / len(idxs_users), 'Global_Acc': cur_g_acc})

        self.test_domain()
        self.logger.info('Training is completed.')
        end_time = time.time()
        self.logger.info(f'Total running time: {end_time - start_time} s || avg: {(end_time - start_time) / self.args.comm_round} s')
        reporter.report()