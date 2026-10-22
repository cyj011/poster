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
class DualGateClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int, dropout: float = 0.3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid(),
        )
        self.stream_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, target_dim),
        )

    def fuse(self, image_features, landmark_features):
        stream_tokens = torch.stack((image_features, landmark_features), dim=1)
        attended_tokens, _ = self.stream_attention(
            stream_tokens,
            stream_tokens,
            stream_tokens,
            need_weights=False,
        )
        attended = self.attention_norm(attended_tokens.mean(dim=1))
        mean_features = (image_features + landmark_features) * 0.5
        gate = self.gate(torch.cat((image_features, landmark_features), dim=-1))
        return gate * attended + (1.0 - gate) * mean_features

    def forward(self, image_features, landmark_features):
        fused = self.fuse(image_features, landmark_features)
        return self.classifier(fused), fused


class pyramid_trans_expr(nn.Module):
    def __init__(self, img_size=224, num_classes=7, type="large", head_type="linear"):
        super().__init__()
        if head_type not in {"linear", "dual-gate"}:
            raise ValueError(f"Unsupported head type: {head_type}")
        self.head_type = head_type

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
                                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)


        if head_type == "linear":
            self.se_block = SE_block(input_dim=512)
            self.head = ClassificationHead(input_dim=512, target_dim=self.num_classes)
        else:
            self.se_block = SE_block(input_dim=512)
            self.head = DualGateClassificationHead(
                input_dim=512,
                target_dim=self.num_classes,
            )

    def train(self, mode=True):
        """Keep the pretrained landmark extractor in inference mode.

        Its weights are frozen, so updating BatchNorm running statistics during
        classifier training would otherwise change the landmark features.
        """
        super().train(mode)
        self.face_landback.eval()
        return self


    def forward(self, x):
        B_ = x.shape[0]
        x_face = F.interpolate(x, size=112)
        _, x_face = self.face_landback(x_face)
        x_face = x_face.view(B_, -1, 49).transpose(1,2)
        ###############  landmark x_face ([B, 49, 512])

        x_ir = self.ir_back(x)
        x_ir = self.ir_layer(x_ir)
        ###############  image x_ir ([B, 49, 512])

        if self.head_type == "dual-gate":
            image_features, landmark_features = self.pyramid_fuse(
                x_ir,
                x_face,
                return_streams=True,
            )
            y_feat = self.se_block(
                self.head.fuse(image_features, landmark_features)
            )
            out = self.head.classifier(y_feat)
        else:
            y_hat = self.pyramid_fuse(x_ir, x_face)
            y_hat = self.se_block(y_hat)
            y_feat = y_hat
            out = self.head(y_hat)

        return out, y_feat
