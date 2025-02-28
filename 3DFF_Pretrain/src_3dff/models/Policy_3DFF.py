from copy import deepcopy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net
from src_3dff.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    CLIPEncoder,
)
from src_3dff.models.policy import ILPolicy
from src_3dff.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
from src_3dff.waypoint_pred.utils import nms
from src_3dff.models.utils import (
    angle_feature_with_ele, dir_angle_feature_with_ele, length2mask, angle_feature_torch, pad_tensors, gen_seq_masks, get_angle_fts, get_angle_feature, get_point_angle_feature, calculate_vp_rel_pos_fts, calc_position_distance,pad_tensors_wgrad, rectangular_to_polar)
import math
from PIL import Image
import cv2
import open3d as o3d
import numpy as np
from src_3dff.models.feature_fields import Feature_Fields


@baseline_registry.register_policy
class Policy_3DFF(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        super().__init__(
            Net_3DFF(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )


class Net_3DFF(Net):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions,
    ):
        super().__init__()

        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        print('\nInitalizing the 3DFF model ...')
        batch_size = 1
        self.feature_fields = Feature_Fields(batch_size=batch_size, device=self.device, mode='habitat',camera_intrinsic=None, bev=True)

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=model_config.spatial_output,
        )
        self.space_pool_depth = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))

        self.rgb_encoder = CLIPEncoder('ViT-L/14@336px',self.device)
        self.space_pool_rgb = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))
        self.pano_img_idxes = np.arange(0, 12, dtype=np.int64)        # anti-closewise
        pano_angle_rad_c = (1-self.pano_img_idxes/12) * 2 * math.pi   # anti-closewise
        self.pano_angle_fts = angle_feature_torch(torch.from_numpy(pano_angle_rad_c))

        self.headings = [0 for i in range(batch_size)]
        self.positions = [0 for i in range(batch_size)]

        self.train()

    @property  # trivial argument, just for init with habitat
    def output_size(self):
        return 1

    @property
    def is_blind(self):
        return self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return 1

    def preprocess_depth(self, depth):
        # depth - (B, H, W, 1) torch Tensor

        min_depth = 0.  # !!!!!!!!!!! This is the setting for R2R
        max_depth = 10. # !!!!!!!!!!! This is the setting for R2R

        # Column-wise post-processing
        depth = depth * 1.0
        H = depth.shape[1]
        depth_max, _ = depth.max(dim=1, keepdim=True)  # (B, H, W, 1)
        depth_max = depth_max.expand(-1, H, -1, -1)
        depth[depth == 0] = depth_max[depth == 0]

        depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0
        depth = depth / 100.
        return depth


    def forward(self, mode=None, 
                txt_ids=None, txt_masks=None, txt_embeds=None, 
                waypoint_predictor=None, observations=None, in_train=True,
                rgb_fts=None, dep_fts=None, loc_fts=None, 
                nav_types=None, view_lens=None,
                gmap_vp_ids=None, gmap_step_ids=None,
                gmap_img_fts=None, gmap_pos_fts=None, 
                gmap_masks=None, gmap_visited_masks=None, gmap_pair_dists=None):
            
        batch_size = len(self.positions)

        ''' encoding rgb/depth at all directions ----------------------------- '''
        NUM_ANGLES = 120    # 120 angles 3 degrees each
        NUM_IMGS = 12
        NUM_CLASSES = 12    # 12 distances at each sector
        depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
        rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)

        # reverse the order of input images to clockwise
        a_count = 0
        for i, (k, v) in enumerate(observations.items()):
            if 'depth' in k:  # You might need to double check the keys order
                for bi in range(v.size(0)):
                    ra_count = (NUM_IMGS - a_count) % NUM_IMGS
                    depth_batch[ra_count + bi*NUM_IMGS] = v[bi]
                    rgb_batch[ra_count + bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi] 
                a_count += 1


        obs_view12 = {}
        obs_view12['depth'] = depth_batch
        obs_view12['rgb'] = rgb_batch

        depth_height = self.feature_fields.args.input_height
        depth_width = self.feature_fields.args.input_width
        layer_width = self.feature_fields.args.mlp_net_width
        depth_batch_fts = torch.zeros((obs_view12['depth'].shape[0],depth_height,depth_width,1))
        for i in range(obs_view12['depth'].shape[0]):
            depth_batch_fts[i] = torch.tensor(cv2.resize(obs_view12['depth'][i].cpu().numpy(), (depth_height, depth_width),  interpolation = cv2.INTER_NEAREST)).view(depth_height, depth_width,1)

        depth_batch_fts = self.preprocess_depth(depth_batch_fts).view(batch_size,12,depth_height*depth_width).numpy()

        with torch.no_grad():
            depth_embedding = self.depth_encoder(obs_view12)  # torch.Size([bs, 128, 4, 4])
            rgb_embedding, grid_batch_fts = self.rgb_encoder(obs_view12)

        grid_batch_fts = grid_batch_fts.view(batch_size,12,depth_height*depth_width,layer_width).cpu().numpy()


        self.feature_fields.update_feature_fields_habitat(self.positions, self.headings, depth_batch_fts, grid_batch_fts)


        ''' waypoint prediction ----------------------------- '''
        waypoint_heatmap_logits = waypoint_predictor(
            rgb_embedding, depth_embedding)

        # reverse the order of images back to counter-clockwise
        rgb_embed_reshape = rgb_embedding.reshape(
            batch_size, NUM_IMGS, 768, 1, 1)
        depth_embed_reshape = depth_embedding.reshape(
            batch_size, NUM_IMGS, 128, 4, 4)
        rgb_feats = torch.cat((
            rgb_embed_reshape[:,0:1,:], 
            torch.flip(rgb_embed_reshape[:,1:,:], [1]),
        ), dim=1)
        depth_feats = torch.cat((
            depth_embed_reshape[:,0:1,:], 
            torch.flip(depth_embed_reshape[:,1:,:], [1]),
        ), dim=1)
        # way_feats = torch.cat((
        #     way_feats[:,0:1,:], 
        #     torch.flip(way_feats[:,1:,:], [1]),
        # ), dim=1)

        # from heatmap to points
        batch_x_norm = torch.softmax(
            waypoint_heatmap_logits.reshape(
                batch_size, NUM_ANGLES*NUM_CLASSES,
            ), dim=1
        )
        batch_x_norm = batch_x_norm.reshape(
            batch_size, NUM_ANGLES, NUM_CLASSES,
        )
        batch_x_norm_wrap = torch.cat((
            batch_x_norm[:,-1:,:], 
            batch_x_norm, 
            batch_x_norm[:,:1,:]), 
            dim=1)
        batch_output_map = nms(
            batch_x_norm_wrap.unsqueeze(1), 
            max_predictions=5,
            sigma=(7.0,5.0))

        # predicted waypoints before sampling
        batch_output_map = batch_output_map.squeeze(1)[:,1:-1,:]       
            
        if in_train:
            # Waypoint augmentation
            # parts of heatmap for sampling (fix offset first)
            HEATMAP_OFFSET = 5
            batch_way_heats_regional = torch.cat(
                (waypoint_heatmap_logits[:,-HEATMAP_OFFSET:,:], 
                waypoint_heatmap_logits[:,:-HEATMAP_OFFSET,:],
            ), dim=1)
            batch_way_heats_regional = batch_way_heats_regional.reshape(batch_size, 12, 10, 12)
            batch_sample_angle_idxes = []
            batch_sample_distance_idxes = []
            # batch_way_log_prob = []
            for j in range(batch_size):
                # angle indexes with candidates
                angle_idxes = batch_output_map[j].nonzero()[:, 0]
                # clockwise image indexes (same as batch_x_norm)
                img_idxes = ((angle_idxes.cpu().numpy()+5) // 10)
                img_idxes[img_idxes==12] = 0
                # # candidate waypoint states
                # way_feats_regional = way_feats[j][img_idxes]
                # heatmap regions for sampling
                way_heats_regional = batch_way_heats_regional[j][img_idxes].view(img_idxes.size, -1)
                way_heats_probs = F.softmax(way_heats_regional, 1)
                probs_c = torch.distributions.Categorical(way_heats_probs)
                way_heats_act = probs_c.sample().detach()
                sample_angle_idxes = []
                sample_distance_idxes = []
                for k, way_act in enumerate(way_heats_act):
                    if img_idxes[k] != 0:
                        angle_pointer = (img_idxes[k] - 1) * 10 + 5
                    else:
                        angle_pointer = 0
                    sample_angle_idxes.append(way_act//12+angle_pointer)
                    sample_distance_idxes.append(way_act%12)
                batch_sample_angle_idxes.append(sample_angle_idxes)
                batch_sample_distance_idxes.append(sample_distance_idxes)
                # batch_way_log_prob.append(
                #     probs_c.log_prob(way_heats_act))
        else:
            # batch_way_log_prob = None
            None

        depth_feats = self.space_pool_depth(depth_feats)
        rgb_feats = self.space_pool_rgb(rgb_feats)

        # for cand
        cand_rgb = []
        cand_depth = []
        cand_angle_fts = []
        cand_img_idxes = []
        cand_angles = []
        cand_distances = []
        for j in range(batch_size):
            if in_train:
                angle_idxes = torch.tensor(batch_sample_angle_idxes[j])
                distance_idxes = torch.tensor(batch_sample_distance_idxes[j])
            else:
                angle_idxes = batch_output_map[j].nonzero()[:, 0]
                distance_idxes = batch_output_map[j].nonzero()[:, 1]
            # for angle & distance
            angle_rad_c = angle_idxes.cpu().float()/120*2*math.pi       # clockwise
            angle_rad_cc = 2*math.pi-angle_idxes.float()/120*2*math.pi  # anti-clockwise
            cand_angle_fts.append( angle_feature_torch(angle_rad_c) )
            cand_angles.append(angle_rad_cc.tolist())
            cand_distances.append( ((distance_idxes + 1)*0.25).tolist() )
            # for img idxes
            img_idxes = 12 - (angle_idxes.cpu().numpy()+5) // 10        # anti-clockwise
            img_idxes[img_idxes==12] = 0
            cand_img_idxes.append(img_idxes)
            # for rgb & depth
            cand_rgb.append(rgb_feats[j, img_idxes, ...])
            cand_depth.append(depth_feats[j, img_idxes, ...])
            
        # for pano
        pano_rgb = rgb_feats                            # B x 12 x 2048
        pano_depth = depth_feats                        # B x 12 x 128
        pano_angle_fts = deepcopy(self.pano_angle_fts)  # 12 x 4
        pano_img_idxes = deepcopy(self.pano_img_idxes)  # 12

        # cand_angle_fts clockwise
        # cand_angles anti-clockwise

        outputs = {
            'cand_rgb': cand_rgb,               # [K x 2048]
            'cand_depth': cand_depth,           # [K x 128]
            'cand_angle_fts': cand_angle_fts,   # [K x 4]
            'cand_img_idxes': cand_img_idxes,   # [K]
            'cand_angles': cand_angles,         # [K]
            'cand_distances': cand_distances,   # [K]

            'pano_rgb': pano_rgb,               # B x 12 x 2048
            'pano_depth': pano_depth,           # B x 12 x 128
            'pano_angle_fts': pano_angle_fts,   # 12 x 4
            'pano_img_idxes': pano_img_idxes,   # 12 
        }
            
        return outputs
  

