# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
from tqdm import tqdm
import numpy as np
import gc

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize, NormalizeImage, PrepareForNet

from utils.util import compute_scale_and_shift, get_interpolate_frames

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8


def _expand_tensor_info(x, ret=[]):
    if isinstance(x, (list, tuple)):
        for xx in x:
            ret += _expand_tensor_info(xx, ret)
    else:
        ret.append((x.dtype, x.shape))

    return ret


def _show_vram(device, name=""):
    max_vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    vram_mb = int(torch.cuda.memory_allocated(device) / (1024 * 1024))
    print(f"[{name}] VRAM Max: {max_vram_mb}MB, Usage: {vram_mb}MB")


class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)

    def _split_get_intermediate_layers(self, split_size, x, idx, return_class_token):
        if x.shape[0] <= split_size:
            return self.pretrained.get_intermediate_layers(
                x,
                self.intermediate_layer_idx[self.encoder],
                return_class_token=True)
        else:
            assert x.shape[0] % split_size == 0
            ret1 = []
            for i in range(0, x.shape[0], split_size):
                ret1.append(
                    self.pretrained.get_intermediate_layers(
                        x[i:i+split_size],
                        self.intermediate_layer_idx[self.encoder],
                        return_class_token=True)
                )
            ret2 = []
            for j in range(len(self.intermediate_layer_idx[self.encoder])):
                ret2.append((
                    torch.cat([ret1[i][j][0] for i in range(len(ret1))], dim=0),
                    torch.cat([ret1[i][j][1] for i in range(len(ret1))], dim=0), # Expect return_class_token=True
                ))
            return ret2

    def forward(self, x):
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14

        if True:
            features = self.pretrained.get_intermediate_layers(x.flatten(0, 1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
        else:
            # NOTE: This split may be unnecessary since there is bigger VRAM usage than this in later processings.
            features = self._split_get_intermediate_layers(4, x.flatten(0, 1), self.intermediate_layer_idx[self.encoder], return_class_token=True)

        # print(_expand_tensor_info(features)) # fp16
        # _show_vram(x.device, "get_intermediate_layers")
        depth = self.head(features, patch_h, patch_w, T) # fp16
        # _show_vram(x.device, "head")
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True).to(depth.dtype) # fp32->fp16
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T)) # return shape [B, T, H, W]

    def chunked_forward(self, x, chunk=8):
        x_shape = x.shape # BTCHW
        features = []
        for xx in x.chunk(chunk, dim=1):
            features.append(self.forward_features(xx.flatten(0, 1)))
        print(len(features), len(features[0]))
    
    def infer_video_depth(self, frames, target_fps, input_size=518, device='cuda', use_amp=True):
        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        frame_list = [frames[i] for i in range(frames.shape[0])]
        frame_step = INFER_LEN - OVERLAP
        org_video_len = len(frame_list)
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len
        
        depth_list = []
        pre_input = None
        for frame_id in tqdm(range(0, org_video_len, frame_step), ncols=80):
            cur_list = []
            for i in range(INFER_LEN):
                cur_list.append(torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0))
            cur_input = torch.cat(cur_list, dim=1).to(device)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            if cur_input.device.type == "cuda":
                with torch.no_grad(), torch.autocast(device_type=cur_input.device.type, enabled=use_amp):
                    depth = self.forward(cur_input) # depth shape: [1, T, H, W] # fp16
            else:
                with torch.no_grad():
                    depth = self.forward(cur_input) # depth shape: [1, T, H, W]

            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
            depth = depth.cpu().float().numpy()
            depth_list += [depth[i][0] for i in range(depth.shape[0])]
            pre_input = cur_input # cur_input.cpu() can reduce 200MB VRAM but it is small size and little slow

        del frame_list
        gc.collect()

        depth_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        for frame_id in range(0, len(depth_list), INFER_LEN):
            if len(depth_list_aligned) == 0:
                depth_list_aligned += depth_list[:INFER_LEN]
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id+kf_id])
            else:
                curr_align = []
                for i in range(len(kf_align_list)):
                    curr_align.append(depth_list[frame_id+i])
                scale, shift = compute_scale_and_shift(np.concatenate(curr_align),
                                                       np.concatenate(ref_align),
                                                       np.concatenate(np.ones_like(ref_align)==1))

                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                post_depth_list = depth_list[frame_id+align_len:frame_id+OVERLAP]
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift
                    post_depth_list[i][post_depth_list[i]<0] = 0
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                for i in range(OVERLAP, INFER_LEN):
                    new_depth = depth_list[frame_id+i] * scale + shift
                    new_depth[new_depth<0] = 0
                    depth_list_aligned.append(new_depth)

                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    new_depth = depth_list[frame_id+kf_id] * scale + shift
                    new_depth[new_depth<0] = 0
                    ref_align.append(new_depth)
            
        depth_list = depth_list_aligned
            
        return np.stack(depth_list[:org_video_len], axis=0), target_fps
