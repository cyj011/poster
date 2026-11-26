import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
from pathlib import Path

from .hyp_crossvit import *
from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone



def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model




class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat


class CosFaceClassificationHead(nn.Module):
    """CosFace head with an angular margin during training only."""

    def __init__(self, input_dim: int, target_dim: int, scale: float = 30.0, margin: float = 0.20):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(target_dim, input_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, x, targets=None):
        x = F.normalize(x.view(x.size(0), -1), dim=1)
        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(x, weight)
        if targets is not None:
            cosine = cosine.clone()
            cosine[torch.arange(x.size(0), device=x.device), targets] -= self.margin
        return cosine * self.scale


class DiDiCMClassificationHead(nn.Module):
    """Compact DiDiCM-inspired posterior refinement head.

    The original DiDiCM models a class posterior with a discrete diffusion
    process. This lightweight variant keeps POSTER's feature extractor intact:
    a linear classifier produces the initial posterior, then a few learned
    denoising steps refine class probabilities conditioned on the feature.
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        steps: int = 3,
        beta_start: float = 0.08,
        beta_end: float = 0.24,
    ):
        super().__init__()
        if steps < 1:
            raise ValueError("DiDiCM steps must be at least 1")
        if not 0.0 < beta_start <= beta_end < 1.0:
            raise ValueError("DiDiCM beta schedule must satisfy 0 < start <= end < 1")
        self.steps = steps
        self.target_dim = target_dim
        self.base = nn.Linear(input_dim, target_dim)
        self.feature_norm = nn.LayerNorm(input_dim)
        self.step_embed = nn.Embedding(steps, input_dim)
        self.denoisers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim + target_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, target_dim),
            )
            for _ in range(steps)
        ])
        betas = torch.linspace(beta_start, beta_end, steps)
        self.register_buffer("betas", betas)

    def forward(self, x):
        features = self.feature_norm(x.view(x.size(0), -1))
        logits = self.base(features)
        posterior = torch.softmax(logits, dim=-1)
        uniform = torch.full_like(posterior, 1.0 / self.target_dim)

        for step, denoiser in enumerate(self.denoisers):
            beta = self.betas[step]
            noisy_posterior = (1.0 - beta) * posterior + beta * uniform
            step_features = features + self.step_embed.weight[step].unsqueeze(0)
            delta = denoiser(torch.cat([step_features, noisy_posterior], dim=-1))
            posterior = torch.softmax(torch.log(noisy_posterior.clamp_min(1e-6)) + delta, dim=-1)

        return torch.log(posterior.clamp_min(1e-6))


class MultiScaleSpatialAdaptor(nn.Module):
    """Lightweight Mamba-Adaptor-style spatial mixer for 7x7 image tokens.

    It keeps the POSTER backbone unchanged and mixes local context with three
    depthwise convolutions at different dilation rates before projecting back
    to the token dimension.
    """

    def __init__(self, input_dim: int = 512, expansion: int = 2):
        super().__init__()
        hidden_dim = input_dim * expansion
        self.norm = nn.LayerNorm(input_dim)
        self.in_proj = nn.Linear(input_dim, hidden_dim * 2)
        self.branches = nn.ModuleList([
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=hidden_dim,
                bias=False,
            )
            for dilation in (1, 2, 3)
        ])
        self.fuse = nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.gate = nn.Linear(input_dim, input_dim)
        self.act = nn.GELU()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = tokens.shape
        if token_count != 49:
            raise ValueError(f"MultiScaleSpatialAdaptor expects 49 tokens, got {token_count}")

        projected = self.in_proj(self.norm(tokens))
        content, gate = projected.chunk(2, dim=-1)
        content = content.transpose(1, 2).reshape(batch_size, -1, 7, 7)
        multi_scale = [self.act(branch(content)) for branch in self.branches]
        mixed = self.act(self.fuse(torch.cat(multi_scale, dim=1)))
        mixed = mixed.flatten(2).transpose(1, 2)
        mixed = mixed * torch.sigmoid(gate)
        channel_gate = torch.sigmoid(self.gate(tokens.mean(dim=1))).unsqueeze(1)
        return self.out_proj(mixed) * channel_gate


class HeatConductionOperator(nn.Module):
    """DCT-based adaptive heat conduction for a fixed spatial token grid.

    Reference: MzeroMiko/vHeat, classification/models/vHeat.py (Heat2D).
    Uses channel-wise frequency decay with softplus instead of ReLU,
    and FP32 spectral math. This is an adapter, not a full vHeat block.
    """

    def __init__(self, dim: int = 512, grid_size: int = 7, fve_dim=None):
        super().__init__()
        fve_dim = dim if fve_dim is None else fve_dim
        if min(dim, grid_size, fve_dim) < 1:
            raise ValueError("HCO dimensions must be positive")
        self.dim = dim
        self.grid_size = grid_size
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)

        self.frequency_embeddings = nn.Parameter(
            torch.zeros(grid_size, grid_size, fve_dim)
        )
        nn.init.trunc_normal_(self.frequency_embeddings, std=0.02)
        self.diffusivity = nn.Linear(fve_dim, dim)
        nn.init.trunc_normal_(self.diffusivity.weight, std=0.02)
        nn.init.constant_(self.diffusivity.bias, -2.0)

        dct = self._make_dct_matrix(grid_size)
        self.register_buffer("dct_matrix", dct, persistent=False)
        freq = torch.arange(grid_size, dtype=torch.float32) * torch.pi / grid_size
        freq_sq = freq.square()
        self.register_buffer(
            "frequency_radius_sq",
            freq_sq[:, None] + freq_sq[None, :],
            persistent=False,
        )

    @staticmethod
    def _make_dct_matrix(size: int) -> torch.Tensor:
        positions = torch.arange(size, dtype=torch.float32)
        frequencies = torch.arange(size, dtype=torch.float32)[:, None]
        matrix = torch.cos(torch.pi * (positions + 0.5) * frequencies / size)
        matrix[0] *= (1.0 / size) ** 0.5
        if size > 1:
            matrix[1:] *= (2.0 / size) ** 0.5
        return matrix

    def _dct2(self, feature_map: torch.Tensor) -> torch.Tensor:
        matrix = self.dct_matrix.to(dtype=feature_map.dtype)
        return torch.matmul(torch.matmul(matrix, feature_map), matrix.t())

    def _idct2(self, frequency_map: torch.Tensor) -> torch.Tensor:
        matrix = self.dct_matrix.to(dtype=frequency_map.dtype)
        return torch.matmul(torch.matmul(matrix.t(), frequency_map), matrix)

    def spectral_decay(self):
        # Non-negative diffusivity prevents amplification of high frequencies.
        k = F.softplus(F.linear(
            self.frequency_embeddings.float(),
            self.diffusivity.weight.float(),
            self.diffusivity.bias.float(),
        ))
        return torch.exp(-k * self.frequency_radius_sq.float().unsqueeze(-1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("HeatConductionOperator expects a [B, N, C] tensor")
        batch_size, token_count, channels = tokens.shape
        expected = self.grid_size * self.grid_size
        if token_count != expected or channels != self.dim:
            raise ValueError(
                f"HeatConductionOperator expects [B, {expected}, {self.dim}], "
                f"got {tuple(tokens.shape)}"
            )

        normalized = self.norm(tokens)
        feature_map = normalized.transpose(1, 2).reshape(
            batch_size, channels, self.grid_size, self.grid_size
        )
        feature_map = self.dwconv(feature_map).permute(0, 2, 3, 1)
        content, gate = self.in_proj(feature_map).chunk(2, dim=-1)

        # Keep the small DCT and exponential in FP32 under CPU/CUDA AMP.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            frequency = self._dct2(content.float().permute(0, 3, 1, 2))
            decay = self.spectral_decay().permute(2, 0, 1).unsqueeze(0)
            conducted = self._idct2(frequency * decay)
        conducted = conducted.permute(0, 2, 3, 1).to(content.dtype)
        output = self.out_proj(self.out_norm(conducted) * F.silu(gate))
        return output.reshape(batch_size, expected, channels)


class ResidualHeatConduction(nn.Module):
    def __init__(self, dim=512, grid_size=7):
        super().__init__()
        self.op = HeatConductionOperator(dim=dim, grid_size=grid_size)
        self.scale = nn.Parameter(torch.full((dim,), 1e-3))

    def forward(self, tokens):
        return tokens + self.scale * self.op(tokens)


class NonCausalSSDAdapter(nn.Module):
    """Compact NC-SSD-style global state mixer for image tokens.

    Each spatial token contributes to one shared state, and each token reads
    from that state. It follows the NC-SSD factorization, without causal scans.
    """

    def __init__(self, dim=512, state_dim=16, local_kernel=3):
        super().__init__()
        if dim < 1 or state_dim < 1 or local_kernel % 2 == 0:
            raise ValueError("dim/state_dim must be positive and local_kernel must be odd")
        self.dim = dim
        self.state_dim = state_dim
        self.norm = nn.LayerNorm(dim)
        self.local_conv = nn.Conv2d(
            dim, dim, kernel_size=local_kernel,
            padding=local_kernel // 2, groups=dim,
        )
        self.to_x = nn.Linear(dim, dim, bias=False)
        self.to_b = nn.Linear(dim, state_dim)
        self.to_c = nn.Linear(dim, state_dim)
        self.to_m = nn.Linear(dim, 1)
        self.to_gate = nn.Linear(dim, dim)
        self.global_readout = nn.Linear(dim, dim, bias=False)
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.full((dim,), 1e-3))

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(f"NonCausalSSDAdapter expects [B, N, {self.dim}]")
        batch, count, channels = tokens.shape
        height = int(count ** 0.5)
        if height * height != count:
            raise ValueError("NonCausalSSDAdapter expects a square token grid")

        normalized = self.norm(tokens)
        local = normalized.transpose(1, 2).reshape(batch, channels, height, height)
        local = self.local_conv(local).flatten(2).transpose(1, 2)
        x = self.to_x(local)
        b = F.softmax(self.to_b(local).float(), dim=1).to(local.dtype)
        c = F.softmax(self.to_c(local).float(), dim=-1).to(local.dtype)
        m = torch.sigmoid(self.to_m(local))

        # NC-SSD: H = (B * m)^T X, Y = C H. The state is shared by all
        # spatial positions, so a changed token can influence every output.
        state = torch.einsum("bln,bld->bnd", b * m, x)
        mixed = torch.einsum("bln,bnd->bld", c, state)
        # Keep an explicit shared-state path. This avoids the global signal
        # being completely removed by the per-token normalization/gate path.
        global_state = state.mean(dim=1, keepdim=True)
        global_update = self.global_readout(global_state).expand(-1, count, -1)
        update = self.out_proj(self.out_norm(mixed) * F.silu(self.to_gate(local)))
        update = update + global_update
        return tokens + self.scale * update


class pyramid_trans_expr(nn.Module):
    def __init__(
        self,
        img_size=224,
        num_classes=7,
        type="large",
        head_type="linear",
        readout_type="original",
        spatial_enhancement="none",
    ):
        super().__init__()
        if head_type not in {"linear", "cosface", "didicm"}:
            raise ValueError(f"Unsupported head type: {head_type}")
        if readout_type not in {"original", "local"}:
            raise ValueError(f"Unsupported readout type: {readout_type}")
        if spatial_enhancement not in {"none", "multiscale", "hco", "ncssd"}:
            raise ValueError(f"Unsupported spatial enhancement: {spatial_enhancement}")
        if spatial_enhancement in {"hco", "ncssd"} and readout_type != "local":
            raise ValueError(
                f"{spatial_enhancement.upper()} requires --readout-type local "
                "to read enhanced image tokens"
            )
        self.head_type = head_type
        self.readout_type = readout_type
        self.spatial_enhancement = spatial_enhancement

        depth = 8
        if type == "small":
            depth = 4
        if type == "base":
            depth = 6
        if type == "large":
            depth = 8

        self.img_size = img_size
        self.num_classes = num_classes

        self.face_landback = MobileFaceNet([112, 112],136)
        pretrain_dir = Path(__file__).resolve().parent / 'pretrain'
        face_landback_checkpoint = torch.load(
            pretrain_dir / 'mobilefacenet_model_best.pth.tar',
            map_location=lambda storage, loc: storage,
        )
        self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])


        for param in self.face_landback.parameters():
            param.requires_grad = False

        ###########################################################################333


        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load(
            pretrain_dir / 'ir50.pth',
            map_location=lambda storage, loc: storage,
        )
        # ir_checkpoint = ir_checkpoint["model"]
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)

        self.ir_layer = nn.Linear(1024,512)

        #############################################################3

        self.pyramid_fuse = HyVisionTransformer(in_chans=49, q_chanel = 49, embed_dim=512,
                                             depth=depth, num_heads=8, mlp_ratio=2.,
                                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                             local_token_readout=readout_type == "local")

        if spatial_enhancement == "multiscale":
            self.spatial_adaptor = MultiScaleSpatialAdaptor(input_dim=512, expansion=2)
            self.spatial_proj = nn.Linear(512, 512, bias=False)
            self.spatial_scale = nn.Parameter(torch.zeros(()))


        self.se_block = SE_block(input_dim=512)
        if head_type == "cosface":
            self.head = CosFaceClassificationHead(
                input_dim=512,
                target_dim=self.num_classes,
            )
        elif head_type == "didicm":
            self.head = DiDiCMClassificationHead(
                input_dim=512,
                target_dim=self.num_classes,
            )
        else:
            self.head = ClassificationHead(input_dim=512, target_dim=self.num_classes)

        # Construct last so shared baseline weights keep the same seeded init.
        if spatial_enhancement == "hco":
            self.hco_adapter = ResidualHeatConduction(dim=512, grid_size=7)
        elif spatial_enhancement == "ncssd":
            self.ncssd_adapter = NonCausalSSDAdapter(dim=512, state_dim=16)

    def train(self, mode=True):
        """Keep the pretrained landmark extractor in inference mode.

        Its weights are frozen, so updating BatchNorm running statistics during
        classifier training would otherwise change the landmark features.
        """
        super().train(mode)
        self.face_landback.eval()
        return self

    def forward(self, x, targets=None):
        B_ = x.shape[0]
        x_face = F.interpolate(x, size=112)
        _, x_face = self.face_landback(x_face)
        x_face = x_face.view(B_, -1, 49).transpose(1,2)
        ###############  landmark x_face ([B, 49, 512])

        x_ir = self.ir_back(x)
        x_ir = self.ir_layer(x_ir)
        ###############  image x_ir ([B, 49, 512])

        need_spatial_tokens = self.spatial_enhancement == "multiscale"
        fused = self.pyramid_fuse(
            x_ir, x_face, return_tokens=need_spatial_tokens,
            image_token_adapter=getattr(
                self, "hco_adapter", getattr(self, "ncssd_adapter", None)
            ),
        )
        if self.spatial_enhancement == "multiscale":
            y_hat, spatial_tokens = fused
            enhanced_tokens = self.spatial_adaptor(spatial_tokens)
            enhanced_feature = enhanced_tokens.mean(dim=1)
            y_hat = y_hat + torch.tanh(self.spatial_scale) * self.spatial_proj(enhanced_feature)
        else:
            y_hat = fused
        y_hat = self.se_block(y_hat)
        y_feat = y_hat
        if self.head_type == "cosface":
            out = self.head(y_hat, targets)
        else:
            out = self.head(y_hat)
        return out, y_feat
