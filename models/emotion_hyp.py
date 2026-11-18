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


class PartPrototypeHead(nn.Module):
    """Class-conditioned online prototype memory over spatial tokens."""

    def __init__(self, input_dim: int, num_classes: int, prototypes_per_class: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.linear = ClassificationHead(input_dim, num_classes)
        self.register_buffer("prototypes", torch.zeros(num_classes, prototypes_per_class, input_dim))
        self.register_buffer("prototype_counts", torch.zeros(num_classes, prototypes_per_class))
        self.register_buffer("prototype_ready", torch.zeros(num_classes, dtype=torch.bool))
        self.prototype_weight = 0.25
        self.prototype_momentum = 0.1
        self.prototype_scale = 10.0

    def forward(self, pooled_features, part_tokens):
        logits = self.linear(pooled_features)
        if not self.prototype_ready.all():
            return logits
        tokens = F.normalize(part_tokens.float(), dim=-1)
        centers = F.normalize(self.prototypes.float(), dim=-1)
        similarities = torch.einsum("bnd,ckd->bnck", tokens, centers)
        # Average a few matching locations to avoid relying on one noisy token.
        part_scores = similarities.topk(min(3, tokens.size(1)), dim=1).values.mean(dim=1).mean(dim=-1)
        return logits + self.prototype_weight * self.prototype_scale * part_scores

    @torch.no_grad()
    def update(self, part_tokens, targets):
        features = F.normalize(part_tokens.detach().float(), dim=-1)
        for class_id in targets.unique().tolist():
            class_features = features[targets == class_id].reshape(-1, features.size(-1))
            if not self.prototype_ready[class_id]:
                # Deterministic farthest-point initialization spreads the centers.
                center = class_features[0]
                nearest_similarity = class_features @ center
                for proto_id in range(self.prototypes_per_class):
                    self.prototypes[class_id, proto_id].copy_(center)
                    nearest_similarity = torch.maximum(
                        nearest_similarity, class_features @ center
                    )
                    center = class_features[nearest_similarity.argmin()]
                self.prototype_counts[class_id].fill_(1.0)
                self.prototype_ready[class_id] = True
                continue

            centers = F.normalize(self.prototypes[class_id].float(), dim=-1)
            assignments = class_features @ centers.t()
            assignments = assignments.argmax(dim=1)
            for proto_id in range(self.prototypes_per_class):
                selected = class_features[assignments == proto_id]
                if selected.numel() == 0:
                    continue
                center = F.normalize(selected.mean(dim=0), dim=0)
                self.prototypes[class_id, proto_id].mul_(1.0 - self.prototype_momentum)
                self.prototypes[class_id, proto_id].add_(center, alpha=self.prototype_momentum)
                self.prototypes[class_id, proto_id].copy_(F.normalize(self.prototypes[class_id, proto_id], dim=0))
                self.prototype_counts[class_id, proto_id] += selected.size(0)


class pyramid_trans_expr(nn.Module):
    def __init__(
        self,
        img_size=224,
        num_classes=7,
        type="large",
        head_type="linear",
        readout_type="original",
    ):
        super().__init__()
        if head_type not in {"linear", "cosface", "part-prototype"}:
            raise ValueError(f"Unsupported head type: {head_type}")
        if readout_type not in {"original", "local"}:
            raise ValueError(f"Unsupported readout type: {readout_type}")
        self.head_type = head_type
        self.readout_type = readout_type

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


        self.se_block = SE_block(input_dim=512)
        if head_type == "cosface":
            self.head = CosFaceClassificationHead(
                input_dim=512,
                target_dim=self.num_classes,
            )
        elif head_type == "part-prototype":
            self.head = PartPrototypeHead(input_dim=512, num_classes=self.num_classes)
        else:
            self.head = ClassificationHead(input_dim=512, target_dim=self.num_classes)

    def train(self, mode=True):
        """Keep the pretrained landmark extractor in inference mode.

        Its weights are frozen, so updating BatchNorm running statistics during
        classifier training would otherwise change the landmark features.
        """
        super().train(mode)
        self.face_landback.eval()
        return self

    def forward(self, x, targets=None, return_parts=False):
        B_ = x.shape[0]
        x_face = F.interpolate(x, size=112)
        _, x_face = self.face_landback(x_face)
        x_face = x_face.view(B_, -1, 49).transpose(1,2)
        ###############  landmark x_face ([B, 49, 512])

        x_ir = self.ir_back(x)
        x_ir = self.ir_layer(x_ir)
        ###############  image x_ir ([B, 49, 512])

        need_parts = self.head_type == "part-prototype"
        fused = self.pyramid_fuse(x_ir, x_face, return_tokens=need_parts)
        if need_parts:
            y_hat, part_tokens = fused
        else:
            y_hat = fused
        y_hat = self.se_block(y_hat)
        y_feat = y_hat
        if self.head_type == "cosface":
            out = self.head(y_hat, targets)
        elif self.head_type == "part-prototype":
            out = self.head(y_hat, part_tokens)
        else:
            out = self.head(y_hat)

        if return_parts:
            if not need_parts:
                raise ValueError("return_parts requires the part-prototype head")
            return out, y_feat, part_tokens.detach()
        return out, y_feat
