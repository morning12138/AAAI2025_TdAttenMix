import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.data.mixup import Mixup, cutmix_bbox_and_lam, one_hot
from timm.models import create_model
from timm.optim import create_optimizer
import numpy as np
import torch
import random
import absvit_control

model_gaze = create_model(
    'absvit_base_patch16_224',
    pretrained=False,
    num_classes=1000,
    drop_rate=0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    pretrained_cfg=None
)
model_gaze.to('cuda')
model_without_ddp = model_gaze
checkpoint = torch.load('./absvit_base_patch16_224.pth')
model_without_ddp.load_state_dict(checkpoint['model'])
_ = model_gaze.eval()


def compute_gaze_saliency(attn):
    cos_sim = F.normalize(attn, dim=-1) @ F.normalize(model_gaze.prompt[None, ..., None], dim=1)  # B, N, 1
    mask = cos_sim.clamp(0, 1)

    attn = mask[:, model_gaze.num_prefix_tokens:].view(mask.shape[0], int(196 ** 0.5), int(196 ** 0.5))
    attn_max = attn.max(dim=1)[0].max(dim=1)[0][..., None, None]
    attn_min = attn.min(dim=1)[0].min(dim=1)[0][..., None, None]
    attn = (attn - (0.7 * attn_max + 0.3 * attn_min)).clamp(0)
    attn_196 = attn.view(attn.shape[0], attn.shape[1] * attn.shape[2])
    return attn_196



def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    mixed_target = y1 * lam + y2 * (1. - lam)
    return mixed_target, y1, y2


def batch_index_generate(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        return idx.reshape(-1)
    elif len(x.size()) == 2:
        B, N = x.size()
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        return idx
    else:
        raise NotImplementedError


class TdAttenMix(Mixup):
    def __init__(self, mixup_alpha=1., prob=1.0, switch_prob=0.5,
                 mode='batch',  label_smoothing=0.1, num_classes=1000, min_side_ratio=0.25, max_side_ratio=0.75, side=14):
        self.mixup_alpha = mixup_alpha
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)
        
        self.side = side
        self.min_side = int(side*min_side_ratio)
        self.max_side = int(side*max_side_ratio)
        self.rectangle_size_list = []
        for i in range(self.min_side, self.max_side+1):
            self.rectangle_size_list.append((i,i))


    def tdattenmix(self, inputs,attn,rectangle_size):
        inputs_side = inputs.shape[2]
        patch_size = inputs_side//self.side
        inputs = torch.nn.functional.unfold(inputs,patch_size,stride=patch_size).transpose(1,2)
        source_image = inputs.flip(0)

        attn = attn.reshape(-1, self.side, self.side).unsqueeze(1)
        rectangle_attn = torch.nn.functional.unfold(attn,rectangle_size,stride=1)
        rectangle_attn = rectangle_attn.sum(dim=1)

        min_region_center_index = torch.argmin(rectangle_attn,dim=1)
        max_region_center_index = torch.argmax(rectangle_attn,dim=1)

        min_region_index = self.index_translate(min_region_center_index,rectangle_size, token_size=(self.side,self.side))
        max_region_index = self.index_translate(max_region_center_index,rectangle_size, token_size=(self.side,self.side))
        min_region_index = batch_index_generate(inputs,min_region_index)
        max_region_index = batch_index_generate(inputs,max_region_index.flip(0))
        
        B,N,C = inputs.shape
        inputs = inputs.reshape(B*N, C)
        source_image = source_image.reshape(B*N, C)
        inputs[min_region_index] = source_image[max_region_index]
        inputs = inputs.reshape(B,N,C)
        inputs = torch.nn.functional.fold(inputs.transpose(1,2),inputs_side,patch_size,stride=patch_size)

        source_mask = torch.zeros_like(attn).bool()
        source_mask = source_mask.reshape(-1)
        source_mask[min_region_index] = True
        source_mask = source_mask.reshape(B,N)
        target_mask = ~source_mask

        max_region_mask = torch.zeros_like(attn).bool()
        max_region_mask = max_region_mask.reshape(-1)
        max_region_mask[max_region_index] = True
        max_region_mask = max_region_mask.reshape(B, N)

        attn = attn.reshape(attn.shape[0], -1)
        attn_source = torch.sum(attn[max_region_mask])
        attn_target = torch.sum(attn[target_mask])
        lam2 = attn_target / (attn_target + attn_source)

        return inputs, target_mask, source_mask, lam2

    def index_translate(self,rectangle_index, rectangle_size=(3,3), token_size=(7,7)):
        total_index = torch.arange(token_size[0]*token_size[1]).reshape(1,1,token_size[0],token_size[1]).cuda()
        total_index_list = torch.nn.functional.unfold(total_index.float(),rectangle_size,stride=1).transpose(1,2).long()
        sequence_index=total_index_list.index_select(dim=1,index=rectangle_index).squeeze(0)
        return sequence_index


    def __call__(self, x, target, motivat_model=None, model_type='other'):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        assert self.mode == 'batch', 'Mixup mode is batch by default'

        use_tdattenmix = np.random.rand() < self.switch_prob
        if use_tdattenmix:
            with torch.no_grad():
                motivat_model.eval()
                if model_type == 'vit':
                    un_mixed_prediction_distribution, _ = motivat_model(x)
                else:
                    un_mixed_prediction_distribution = motivat_model(x)
                motivat_model.train()
                model_gaze.eval()
                _, attn = model_gaze(x, 1, y=target)
            rectangle_size = random.choice(self.rectangle_size_list)
            # Following the original Mixup code of Timm codebase, lam indicates the area ratio of target image, which is equal to the (1-\lambda) in the paper.
            lam1 = (self.side**2-rectangle_size[0]*rectangle_size[1])/self.side**2
            x, target_mask, source_mask, lam2 = self.tdattenmix(x, attn, rectangle_size)

            lam = lam1 * 0.5 + lam2 * 0.5
            mixed_target, target_target, source_target = mixup_target(target, self.num_classes, lam, self.label_smoothing, x.device) # tuple or tensor
            target_prediction_distribution = lam*un_mixed_prediction_distribution
            source_prediction_distribution = (1-lam)*un_mixed_prediction_distribution.flip(0)
            mixed_prediction_distribution = target_prediction_distribution + source_prediction_distribution
            
            return x, mixed_target, (target_target, source_target, target_mask, source_mask, mixed_prediction_distribution)

        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            if not lam == 1:
                lam = float(lam)
                x_flipped = x.flip(0).mul_(1. - lam)
                x.mul_(lam).add_(x_flipped)
            mixed_target, _, _ = mixup_target(target, self.num_classes, lam, self.label_smoothing, x.device) # tuple or tensor

            return x, mixed_target, None
        
        


