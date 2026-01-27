import torch
from typing import Dict, List
import torchvision.models as models


class OnlineSharpeningStats:
    """
    Online GPU statistics for representational sharpening.
    Mixed precision:
      - sum, sum_norm: bf16
      - sumsq: fp32
    """

    def __init__(
        self,
        layer_names: List[str],
        num_classes: int,
        device: torch.device = torch.device("cuda"),
        eps: float = 1e-8,
    ):
        self.layer_names = layer_names
        self.C = int(num_classes)
        self.device = device
        self.eps = eps

        # lazy init per layer
        self._dim = {}
        self.count = {}
        self.sum = {}
        self.sum_norm = {}
        self.sumsq = {}

    def _init_layer(self, layer: str, D: int):
        self._dim[layer] = D

        self.count[layer] = torch.zeros(
            self.C, device=self.device, dtype=torch.long
        )
        self.sum[layer] = torch.zeros(
            self.C, D, device=self.device, dtype=torch.bfloat16
        )
        self.sum_norm[layer] = torch.zeros(
            self.C, D, device=self.device, dtype=torch.bfloat16
        )
        self.sumsq[layer] = torch.zeros(
            self.C, D, device=self.device, dtype=torch.float32
        )

    @torch.no_grad()
    def update(self, feats: Dict[str, torch.Tensor], labels: torch.Tensor):
        """
        feats: {layer_name: flattened CUDA tensor [D] or [B, D]}
        labels: CUDA or CPU tensor [B] or scalar, class indices
        """
        if labels.dim() == 0:
            labels = labels.view(1)
        labels = labels.to(self.device).long()
        B = labels.numel()

        for layer, x in feats.items():
            if layer not in self.layer_names:
                continue

            if x.dim() == 1:
                x = x.unsqueeze(0)
            assert x.size(0) == B

            D = x.size(1)
            if layer not in self._dim:
                self._init_layer(layer, D)
            else:
                if self._dim[layer] != D:
                    raise RuntimeError(f"Dim mismatch for layer {layer}")

            # ---- counts ----
            bc = torch.bincount(labels, minlength=self.C)
            self.count[layer] += bc

            # ---- sum (bf16) ----
            x_bf16 = x.to(torch.bfloat16)
            self.sum[layer].index_add_(0, labels, x_bf16)

            # ---- sumsq (fp32, stable) ----
            x_fp32 = x.float()
            self.sumsq[layer].index_add_(0, labels, x_fp32 * x_fp32)

            # ---- sum of normalized vectors (bf16) ----
            norms = torch.linalg.norm(x_fp32, dim=1, keepdim=True).clamp_min_(self.eps)
            x_norm = (x_fp32 / norms).to(torch.bfloat16)
            self.sum_norm[layer].index_add_(0, labels, x_norm)

    @torch.no_grad()
    def finalize(self):
        """
        Returns per-layer dict with:
          - count            [C]
          - trace_var        [C]
          - within_cosine    [C]
        """
        results = {}

        for layer in self.layer_names:
            n = self.count[layer]                    # [C]
            n_f = n.float().clamp_min(1.0)           # avoid div0
            denom = n_f.unsqueeze(1)

            sum_fp32 = self.sum[layer].float()
            mu = sum_fp32 / denom                    # [C, D]

            ex2 = self.sumsq[layer] / denom
            var_diag = (ex2 - mu * mu).clamp_min_(0.0)
            trace_var = var_diag.sum(dim=1)           # [C]

            # within-class mean cosine similarity
            sn = self.sum_norm[layer].float()
            sn_norm2 = (sn * sn).sum(dim=1)
            within = (sn_norm2 - n_f) / (n_f * (n_f - 1.0)).clamp_min(1.0)
            within = torch.where(n >= 2, within, torch.nan * within)

            results[layer] = {
                "count": n,
                "trace_var": trace_var,
                "within_cosine": within,
            }

        return results

    @torch.no_grad()
    def fisher_pairwise(self, layer: str, eps: float = 1e-12):
        """
        Optional: Fisher-style class separability matrix [C, C]
        J_ij = ||mu_i - mu_j||^2 / (trace_i + trace_j)
        """
        stats = self.finalize()[layer]
        n = self.count[layer].float().clamp_min(1.0)

        mu = self.sum[layer].float() / n.unsqueeze(1)
        tr = stats["trace_var"]

        mu2 = (mu * mu).sum(dim=1)
        dist2 = mu2[:, None] + mu2[None, :] - 2 * (mu @ mu.T)

        denom = (tr[:, None] + tr[None, :]).clamp_min(eps)
        return dist2 / denom


if __name__ == '__main__':
    device = torch.device("cuda")
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    model.eval()

    # ===== 3) 建在线统计器 =====
    stats = OnlineSharpeningStats(
        layer_names=["conv4_3", "conv5_3", "fc1", "fc2"],
        num_classes=1000,
        device=device,
    )

    # ===== 4) 注册 forward hooks =====
    # VGG16 features 的关键 index（torchvision 标准实现）
    # conv4_3 的 ReLU 输出是 features[22]（conv 在 21，relu 在 22）
    # conv5_3 的 ReLU 输出是 features[29]（conv 在 28，relu 在 29）
    # fc1 ReLU 输出是 classifier[1]（Linear 在 0，ReLU 在 1）
    # fc2 ReLU 输出是 classifier[4]（Linear 在 3，ReLU 在 4）
    layer_map = {
        "conv4_3": model.features[22],
        "conv5_3": model.features[29],
        "fc1":     model.classifier[1],
        "fc2":     model.classifier[4],
    }

    # 用于每次 forward 临时存 activations
    _cache = {}

    def make_hook(name: str):
        def hook(module, inp, out):
            # out is on GPU; detach to avoid graph refs
            _cache[name] = out.detach()
        return hook

    hooks = []
    for name, mod in layer_map.items():
        hooks.append(mod.register_forward_hook(make_hook(name)))

    # ===== 5) 一个“单步处理”的函数：forward -> 抓激活 -> update统计量 =====
    @torch.no_grad()
    def process_one(img: torch.Tensor, label: torch.Tensor):
        """
        img: [1,3,224,224] cuda
        label: scalar or [1] (cpu/cuda都行)
        """
        _cache.clear()

        _ = model(img)  # forward; hooks fill _cache

        # 把抓到的激活 flatten 成 [B,D]（这里 B=1）
        feats = {}
        for k in ["conv4_3", "conv5_3", "fc1", "fc2"]:
            x = _cache[k]
            if x.dim() > 2:
                # conv: [B,C,H,W] -> [B, C*H*W]
                x = x.flatten(start_dim=1)
            else:
                # fc ReLU: [B,4096] already
                # 若是 [4096]，也处理成 [1,4096]
                if x.dim() == 1:
                    x = x.unsqueeze(0)
            feats[k] = x

        stats.update(feats=feats, labels=label)

    # ===== 6) 示例：遍历数据集（伪代码） =====
    # 你只要保证 dataloader 每次给你 batch=1 即可
    # for img_cpu, label_cpu in dataloader:
    #     img = img_cpu.to(device, non_blocking=True)      # [1,3,224,224]
    #     label = label_cpu.to(device, non_blocking=True)  # scalar or [1]
    #     process_one(img, label)

    # ===== 7) 最后拿结果 =====
    # out = stats.finalize()
    # print(out["conv4_3"]["trace_var"].shape)      # [1000]
    # print(out["conv4_3"]["within_cosine"].shape)  # [1000]

    # ===== 8) 清理 hooks（很重要） =====
    def remove_hooks():
        for h in hooks:
            h.remove()