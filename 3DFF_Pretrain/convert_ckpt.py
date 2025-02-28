from habitat import Config
import torch

ckpt = torch.load("data/logs/checkpoints/release_3dff/ckpt.iter50000.pth") # Input the checkpoint pre-trained from 3DFF model
ckpt = ckpt['state_dict']
new_ckpt = {}
for key in ckpt:
    if "net.module.feature_fields." in key:
        new_key = key[len("net.module.feature_fields."):]
        new_ckpt[new_key] = ckpt[key]
    elif "net.feature_fields." in key:
        new_key = key[len("net.feature_fields."):]
        new_ckpt[new_key] = ckpt[key]

torch.save(new_ckpt,"3dff.pth") # Save the checkpoint for downstream tasks
