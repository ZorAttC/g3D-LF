import os
import sys
import random
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from src_3dff.common.base_il_trainer import BaseVLNCETrainer
from src_3dff.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from src_3dff.common.utils import extract_instruction_tokens
from src_3dff.utils import reduce_loss
from src_3dff.models.utils import get_angle_fts


from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from src_3dff.common.utils import dis_to_con, gather_list_and_concat
from habitat_extensions.measures import NDTW, StepsTaken

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from src_3dff.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence
import habitat_sim
import cv2
from PIL import Image
import clip
import open3d as o3d
import re
import matplotlib.pyplot as plt
import matplotlib
simulator_episodes = 0

def focal_loss(inputs, targets, focal_rate=0.1):
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    focal_num = max(int(focal_rate * targets.shape[-1]),1)
    focal_loss = ce_loss.mean() + torch.topk(ce_loss.view(-1),focal_num)[0].mean()
    return focal_loss


@baseline_registry.register_trainer(name="SS-ETP")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        self.config = config

        # load the category embeddings
        category_data = torch.load("data/SceneVerse/category_embeddings.pth")
        self.category_dict, self.category_embeddings = category_data[0], category_data[1]
        self.category_embeddings = self.category_embeddings / torch.linalg.norm(self.category_embeddings, dim=-1, keepdim=True)
        # Load the data of HM3D
        '''
        instance_id_to_label_list = os.listdir("data/SceneVerse/HM3D/scan_data/instance_id_to_label")
        self.hm3d_instance_id_to_label = {}
        for file_name in instance_id_to_label_list:
            if file_name[6:17] in self.hm3d_instance_id_to_label:
                self.hm3d_instance_id_to_label[file_name[6:17]].append("data/SceneVerse/HM3D/scan_data/instance_id_to_label/"+file_name)
            else:
                self.hm3d_instance_id_to_label[file_name[6:17]] = ["data/SceneVerse/HM3D/scan_data/instance_id_to_label/"+file_name]
        
        pcd_with_global_alignment_list = os.listdir("data/SceneVerse/HM3D/scan_data/pcd_with_global_alignment") # use the preprocessed pcd with labels, not of sceneverse
        self.hm3d_pcd_with_global_alignment = {}
        for file_name in pcd_with_global_alignment_list:
            if file_name[6:17] in self.hm3d_pcd_with_global_alignment:
                self.hm3d_pcd_with_global_alignment[file_name[6:17]].append("data/SceneVerse/HM3D/scan_data/pcd_with_global_alignment/"+file_name)
            else:
                self.hm3d_pcd_with_global_alignment[file_name[6:17]] = ["data/SceneVerse/HM3D/scan_data/pcd_with_global_alignment/"+file_name]
        
        '''
        self.hm3d_language_annotations = json.load(open("data/SceneVerse/HM3D/annotations/3dff_hm3d_annotations.json","r"))
        
        pcd_with_global_alignment_list = os.listdir("data/scene_datasets/hm3d/hm3d-train-semantic-annots-v0.2") # use the preprocessed pcd with labels, not of sceneverse
        self.hm3d_pcd_with_global_alignment = {}
        for file_name in pcd_with_global_alignment_list:
            if file_name[6:17] in self.hm3d_pcd_with_global_alignment:
                self.hm3d_pcd_with_global_alignment[file_name[6:17]].append("data/scene_datasets/hm3d/hm3d-train-semantic-annots-v0.2/"+file_name+"/"+file_name[6:17]+".semantic.pth")
            else:
                self.hm3d_pcd_with_global_alignment[file_name[6:17]] = ["data/scene_datasets/hm3d/hm3d-train-semantic-annots-v0.2/"+file_name+"/"+file_name[6:17]+".semantic.pth"]

        # Load the data of Structured3D
        self.structure_3d_scenes = {}
        for i in range(3500):
            if (1600 <= i and i <= 1799) or i in [1155,1816,1913,2034,3499]: # data was lost, see https://github.com/bertjiazheng/Structured3D/issues/30
                continue
            scene_path = 'data/Structured3D/scene_'+str(i).rjust(5, "0")+'/2D_rendering/' 
            self.structure_3d_scenes['scene_'+str(i).rjust(5, "0")] = scene_path

        '''
        instance_id_to_label_list = os.listdir("data/SceneVerse/Structured3D/scan_data/instance_id_to_label")
        self.Structured3D_instance_id_to_label = {}
        for file_name in instance_id_to_label_list:
            i = int(file_name[6:11])
            if (1600 <= i and i <= 1799) or i in [1155,1816,1913,2034,3499]: # data was lost, see https://github.com/bertjiazheng/Structured3D/issues/30
                continue
            if file_name[:11] in self.Structured3D_instance_id_to_label:
                self.Structured3D_instance_id_to_label[file_name[:11]].append("data/SceneVerse/Structured3D/scan_data/instance_id_to_label/"+file_name)
            else:
                self.Structured3D_instance_id_to_label[file_name[:11]] = ["data/SceneVerse/Structured3D/scan_data/instance_id_to_label/"+file_name]

        pcd_with_global_alignment_list = os.listdir("data/SceneVerse/Structured3D/scan_data/pcd_with_global_alignment")
        self.Structured3D_pcd_with_global_alignment = {}
        for file_name in pcd_with_global_alignment_list:
            i = int(file_name[6:11])
            if (1600 <= i and i <= 1799) or i in [1155,1816,1913,2034,3499]: # data was lost, see https://github.com/bertjiazheng/Structured3D/issues/30
                continue
            if file_name[:11] in self.Structured3D_pcd_with_global_alignment:
                self.Structured3D_pcd_with_global_alignment[file_name[:11]].append("data/SceneVerse/Structured3D/scan_data/pcd_with_global_alignment/"+file_name)
            else:
                self.Structured3D_pcd_with_global_alignment[file_name[:11]] = ["data/SceneVerse/Structured3D/scan_data/pcd_with_global_alignment/"+file_name]

        self.Structured3D_language_annotations = json.load(open("data/SceneVerse/Structured3D/annotations/3dff_structured3d_annotations.json","r"))
        '''

        # Load the data of ScanNet
        self.scannet_3d_scenes = {}
        path = 'data/ScanNet/scannet_train_images/frames_square/'
        scenes = json.load(open('data/ScanNet/scannetv2_train_sort.json','r')) # only load train split of scannet
        for scene_id in scenes:    
            self.scannet_3d_scenes[scene_id] = path+scene_id

        instance_id_to_label_list = os.listdir("data/SceneVerse/ScanNet/scan_data/instance_id_to_label")
        self.ScanNet_instance_id_to_label = {}
        for file_name in instance_id_to_label_list:
            if file_name[:12] not in self.scannet_3d_scenes: # only load train split of scannet
                continue
            if file_name[:12] in self.ScanNet_instance_id_to_label:
                self.ScanNet_instance_id_to_label[file_name[:12]].append("data/SceneVerse/ScanNet/scan_data/instance_id_to_label/"+file_name)
            else:
                self.ScanNet_instance_id_to_label[file_name[:12]] = ["data/SceneVerse/ScanNet/scan_data/instance_id_to_label/"+file_name]

        pcd_with_global_alignment_list = os.listdir("data/SceneVerse/ScanNet/scan_data/pcd_with_global_alignment")
        self.ScanNet_pcd_with_global_alignment = {}
        for file_name in pcd_with_global_alignment_list:
            if file_name[:12] not in self.scannet_3d_scenes: # only load train split of scannet
                continue
            if file_name[:12] in self.ScanNet_pcd_with_global_alignment:
                self.ScanNet_pcd_with_global_alignment[file_name[:12]].append("data/SceneVerse/ScanNet/scan_data/pcd_with_global_alignment/"+file_name)
            else:
                self.ScanNet_pcd_with_global_alignment[file_name[:12]] = ["data/SceneVerse/ScanNet/scan_data/pcd_with_global_alignment/"+file_name]

        self.ScanNet_language_annotations = json.load(open("data/SceneVerse/ScanNet/annotations/3dff_scannet_annotations.json","r"))
        self.scannet_align_matrix = json.load(open("data/ScanNet/scannet_align_matrix.json","r"))

    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int):
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                "iteration": iteration,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
        )

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            #H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    #camera_config.WIDTH = H
                    #camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    #camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size

        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ):
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from src_3dff.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov79' if self.config.MODEL.task_type == 'rxr' else 'data/wp_pred/check_cwp_bestdist_hfov90'
        self.waypoint_predictor.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict'],strict=False)
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, find_unused_parameters=False, broadcast_buffers=False)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.policy.parameters()), lr=self.config.IL.lr, eps=1e-5)

        if load_from_ckpt:
            if config.IL.is_requeue:
                import glob
                ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")) )
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1]
            else:
                ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            start_iter = ckpt_dict["iteration"]

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                self.policy.load_state_dict(ckpt_dict["state_dict"],strict=False)
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
            else:
                self.policy.load_state_dict(ckpt_dict["state_dict"],strict=False)
            if config.IL.is_requeue:
                try:
                    self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                except:
                    print("Optim_state is not loaded")

            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")

        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + start_iter
        self.config.freeze()
        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)
        return start_iter


    def _teacher_action(self, batch_angles, batch_distances):
        cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
        oracle_cand_idx = []
        for j in range(len(batch_angles)):
            for k in range(len(batch_angles[j])):
                angle_k = batch_angles[j][k]
                forward_k = batch_distances[j][k]
                dist_k = self.envs.call_at(j, "cand_line_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                cand_dists_to_goal[j].append(dist_k)
            curr_dist_to_goal = self.envs.call_at(j, "current_line_dist_to_goal")
        # if within target range (which def as 3.0)
            if curr_dist_to_goal < 1.5:
                oracle_cand_idx.append(-1)
            else:
                oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))

        return oracle_cand_idx


    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            for k, v in batch.items():
                batch[k] = v[state_index]

        return envs, batch

    def train(self):
        self._set_config()
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))

        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters
        log_every  = self.config.IL.log_every
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.scaler = GradScaler()
        logger.info('Training Starts... GOOD LUCK!')
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0))
            cur_iter = idx + interval
    
            logs = self._train_interval(interval, self.config.IL.ml_weight)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                logger.info(loss_str)
                self.save_checkpoint(cur_iter)

        
    def _train_interval(self, interval, ml_weight):
        self.policy.train()
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()

        self.waypoint_predictor.eval()
        self.category_embeddings = self.category_embeddings.to(self.device)

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        for idx in pbar:
            self.optimizer.zero_grad()
            self.loss = 0.

            with autocast():
                self.rollout('train', ml_weight)

            if self.loss != 0.:
                self.scaler.scale(self.loss).backward() # self.loss.backward()
                for parms in self.policy.net.parameters():
                    if parms.grad != None and torch.any(torch.isnan(parms.grad)):
                        parms.grad[torch.isnan(parms.grad)] = 0

                nn.utils.clip_grad_norm_(self.policy.net.parameters(), max_norm=5.)
                self.scaler.step(self.optimizer) # self.optimizer.step()
                self.scaler.update()

            gc.collect()
            torch.cuda.empty_cache()
            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
            
        return deepcopy(self.logs)
           

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.IL.ckpt_to_load = checkpoint_path
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            #H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    #camera_config.WIDTH = H
                    #camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    #camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj[::5] if self.config.EVAL.fast_eval else self.traj,
            auto_reset_done=False, # unseen: 11006 
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None

        while len(self.stat_eps) < eps_to_eval:
            self.rollout('eval')
        self.envs.close()

        if self.world_size > 1:
            distr.barrier()


    def sim_matrix_cross_entropy(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

    def contrastive_loss(self, fts_1, fts_2, logit_scale=10.):
        sim_matrix = logit_scale * torch.matmul(fts_1, fts_2.t())
        sim_loss1 = self.sim_matrix_cross_entropy(sim_matrix)
        sim_loss2 = self.sim_matrix_cross_entropy(sim_matrix.T)
        sim_loss = (sim_loss1 + sim_loss2)
        return sim_loss


    def fine_grained_contrastive_loss(self, batch_fts_1, batch_fts_2, logit_scale=100.):
        batch_fts_1 = batch_fts_1 / (torch.linalg.norm(batch_fts_1, dim=-1, keepdim=True) + 1e-7)
        batch_sim_score = []
        for batch_id in range(len(batch_fts_2)):
            fts_2 = batch_fts_2[batch_id]
            fts_2 = fts_2[torch.abs(fts_2).sum(-1) != 0]
            fts_2_length = fts_2.shape[0]
            fts_2 = fts_2 / torch.linalg.norm(fts_2, dim=-1, keepdim=True)
            sim_matrix = logit_scale * torch.matmul(batch_fts_1, fts_2.t())
            sim_matrix = sim_matrix.view(batch_fts_1.shape[0],-1)
            sim_score =  torch.topk(sim_matrix,fts_2_length, dim=-1)[0].mean(dim=-1).view(1,-1)
            batch_sim_score.append(sim_score)
        batch_sim_score = torch.cat(batch_sim_score,dim=0)

        sim_loss1 = self.sim_matrix_cross_entropy(sim_matrix)
        sim_loss2 = self.sim_matrix_cross_entropy(sim_matrix.T)
        sim_loss = (sim_loss1 + sim_loss2)
        return sim_loss


    def parse_camera_info(self, camera_info, height, width):
        """ extract intrinsic and extrinsic matrix
        """
        lookat = camera_info[3:6] / np.linalg.norm(camera_info[3:6])
        up = camera_info[6:9] / np.linalg.norm(camera_info[6:9])

        W = lookat
        U = np.cross(W, up)
        V = np.cross(W, U)

        rot = np.vstack((U, V, W))
        trans = camera_info[:3] / 1000.

        xfov = camera_info[9]
        yfov = camera_info[10]

        K = np.diag([1, 1, 1])

        K[0, 2] = width / 2
        K[1, 2] = height / 2

        K[0, 0] = K[0, 2] / np.tan(xfov)
        K[1, 1] = K[1, 2] / np.tan(yfov)

        return rot, trans, K

    def run_on_hm3d(self, mode):

        global simulator_episodes
        simulator_episodes += 1
        if simulator_episodes % 100 == 0:
            self.envs.close()
            self.envs = construct_envs(
                self.config, 
                get_env_class(self.config.ENV_NAME),
                auto_reset_done=False
            )

        loss = 0.
        loss_1 = loss_2 = loss_3 = loss_4 = loss_5 = loss_6 = 0.
        total_actions = 0.
        self.envs.resume_all()
        observations = self.envs.reset()
        instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0

        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                    max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        
        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.stat_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            
        batch_size = len(observations)

        # Get the instance label
        batch_scene_id = []
        batch_pcd_xyz = []
        batch_pcd_label = []
        #batch_instance_id_to_label = []
        batch_instance_id_to_object_type = []
        for b in range(batch_size):
            pcd_xyz = []
            pcd_label = []
            scene_id = self.envs.current_episodes()[b].scene_id.split('/')[-1][:-10]
            batch_scene_id.append(scene_id)
            if scene_id in self.hm3d_pcd_with_global_alignment:
                pcd_file_list = self.hm3d_pcd_with_global_alignment[scene_id]
                for pcd_file in pcd_file_list:
                    pcd_file = torch.load(pcd_file)
                    pcd_xyz.append(torch.tensor(pcd_file[0]))
                    pcd_label.append(torch.tensor(pcd_file[-1]))

                batch_instance_id_to_object_type.append(pcd_file[1])
                batch_pcd_xyz.append(torch.cat(pcd_xyz,0))
                batch_pcd_label.append(torch.cat(pcd_label,0))

                #instance_id_to_label_list = self.hm3d_instance_id_to_label[scene_id]
                #label_dict = {}
                #for label_file in instance_id_to_label_list:
                #    label_file = torch.load(label_file)
                #    label_dict.update(label_file)
                #batch_instance_id_to_label.append(label_dict)
            else:
                batch_pcd_xyz.append(None)
                batch_pcd_label.append(None)
                #batch_instance_id_to_label.append(None)
                batch_instance_id_to_object_type.append(None)
            
        num_of_annotated_scenes = 0
        for scene_id in batch_scene_id:
            if scene_id in self.hm3d_pcd_with_global_alignment:
                num_of_annotated_scenes += 1


        # Store features, for contrastive learning
        sampled_predicted_view_fts = []
        sampled_gt_view_fts = []
        sampled_predicted_panorama_fts = []
        sampled_gt_panorama_fts = []

        sampled_predicted_region_fts = []
        sampled_gt_region = []
        sampled_predicted_panorama_for_text_align = []       
        sampled_gt_panorama_for_text_align = []

        sampled_predicted_panorama_for_fine_grained_text_align = []       
        sampled_gt_panorama_for_fine_grained_text_align = []

        for stepk in range(self.max_len): 

            total_actions += 1
            # Store features and calculate loss at each step, for saving memory
            sampled_predicted_bev_fts = []
            sampled_gt_bev = []
            sampled_predicted_bev_for_text_align = []
            sampled_gt_bev_for_text_align = []

            sampled_predicted_bev_for_fine_grained_text_align = []       
            sampled_gt_bev_for_fine_grained_text_align = []

            positions = []; headings = []
            for ob_i in range(len(observations)):
                agent_state_i = self.envs.call_at(ob_i,
                        "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])

            
            policy_net = self.policy.net
            if hasattr(self.policy.net, 'module'):
                policy_net = self.policy.net.module

            if stepk == 0:
                policy_net.feature_fields.reset(batch_size, batch_gt_pcd_xyz=batch_pcd_xyz, batch_gt_pcd_label=batch_pcd_label) # Reset the settings of 3D feature fields

            batch_size = len(observations)

            policy_net.positions = positions
            policy_net.headings = [(heading+2*math.pi)%(2*math.pi) for heading in headings]

            # cand waypoint prediction
            wp_outputs = self.policy.net(
                waypoint_predictor = self.waypoint_predictor,
                observations = batch,
                in_train = (mode == 'train' and self.config.IL.waypoint_aug),
            )
            batch_angles, batch_distances = wp_outputs['cand_angles'], wp_outputs['cand_distances']
            
            if mode == 'train':

                sampled_view_num = 3 # The number of sampled novel views for each batch_id
                sampled_bev_num = 1 # The number of sampled bev maps for each batch_id

                for sampled_id in range(sampled_view_num):
                    batch_selected_heading_angle = []
                    batch_selected_position = []
                    batch_selected_view_rgb = []
                    #batch_selected_view_depth = []
                    for b in range(batch_size):                
                        selected_nodes = random.choices(list(range(len(batch_angles[b]))), k=1)
                        node_id = selected_nodes[0]

                        selected_position = self.envs.call_at(b, "get_cand_real_pos", {"angle": batch_angles[b][node_id], "forward": batch_distances[b][node_id]})
                        selected_heading_angle = random.uniform(-math.pi,math.pi)
                        q1 = math.cos(selected_heading_angle/2)
                        q2 = math.sin(selected_heading_angle/2)
                        selected_rotation = np.quaternion(q1,0,q2,0)

                        camera_obs = self.envs.call_at(b, "get_observation",{"source_position":selected_position,"source_rotation":selected_rotation})

                        view_rgb = torch.tensor(camera_obs['rgb']).to(self.device).unsqueeze(0)
                        #view_depth = torch.tensor(camera_obs['depth']).to(self.device).unsqueeze(0)

                        batch_selected_heading_angle.append(selected_heading_angle)
                        batch_selected_position.append(selected_position)
                        batch_selected_view_rgb.append(view_rgb)
                        #batch_selected_view_depth.append(view_depth)


                    # Get ground truth novel view CLIP features
                    rgb_input = {}
                    rgb_input['rgb'] = torch.cat(batch_selected_view_rgb,dim=0)
                    with torch.no_grad():
                        clip_fts, _ = policy_net.rgb_encoder(rgb_input)   
                    sampled_gt_view_fts.append(clip_fts)

                    # Predict the novel view features and the instance id within this view
                    predicted_view_fts, predicted_region_fts, predicted_feature_map, gt_label = policy_net.feature_fields.run_view_encode(batch_selected_position, batch_selected_heading_angle,visualization=False)
                    sampled_predicted_view_fts.append(predicted_view_fts)

                    # Get language supervision of region features

                    for batch_id in range(batch_size):
                        if gt_label[batch_id] != None:
                            #instance_id_to_label = batch_instance_id_to_label[batch_id]
                            selected_mask = (gt_label[batch_id]>0).view(-1)
                            selected_gt_label = gt_label[batch_id][selected_mask]
                            if selected_gt_label.shape[0] == 0:
                                continue

                            selected_region_fts = predicted_feature_map[batch_id][selected_mask]

                            #count = 0
                            #output_str = ''
                            for ray_id in range(selected_gt_label.shape[0]):
                                #count += 1
                                instance_id = int(selected_gt_label[ray_id].item())
                                category_id = batch_instance_id_to_object_type[batch_id][instance_id][1] #instance_id_to_label[instance_id]
                                #output_str += " "+category_id + " "
                                if category_id in self.category_dict:
                                    category_embedding_index = self.category_dict[category_id]
                                    sampled_gt_region.append(category_embedding_index)
                                    sampled_predicted_region_fts.append(selected_region_fts[ray_id:ray_id+1])
                                #else:
                                    #output_str += "       "
                                #if count%12-1==0:
                                #    output_str += '\n'
                            #print(output_str)

                    #import matplotlib.pyplot as plt
                    #plt.figure("Image")
                    #plt.imshow(camera_obs['rgb'])
                    #plt.title('image') 
                    #plt.show()

                    # Predict the panorama features and the instance id within this panorama
                    predicted_panorama_fts, shuffled_view_id = policy_net.feature_fields.run_panorama_encode(batch_selected_position,batch_selected_heading_angle,view_shuffle=True,visualization=False)
                    sampled_predicted_panorama_fts.append(predicted_panorama_fts[:,shuffled_view_id]) # Only store the shuffled view (forward-facing)
                    sampled_gt_panorama_fts.append(clip_fts)

                    predicted_panorama_fts = predicted_panorama_fts / torch.linalg.norm(predicted_panorama_fts, dim=-1, keepdim=True)
                    
                    # Get language supervision of panorama features
                    fine_grained_sample_num = 0
                    for batch_id in range(batch_size):
                        if gt_label[batch_id] != None:
                            selected_mask = (gt_label[batch_id]>0).view(-1)
                            selected_gt_label = gt_label[batch_id][selected_mask]
                            if selected_gt_label.shape[0] == 0:
                                continue

                            selected_gt_label = list(set(selected_gt_label.squeeze(1).cpu().numpy().tolist()))
                            random.shuffle(selected_gt_label)
                            # Text for panorama
                            scene_id = batch_scene_id[batch_id]
                            for selected_instance in selected_gt_label:

                                sceneverse_id = batch_instance_id_to_object_type[batch_id][selected_instance][0] # Convert HM3D_id to Sceneverse_id
                                
                                if str(sceneverse_id) in self.hm3d_language_annotations[scene_id]:
                                    text_list = self.hm3d_language_annotations[scene_id][str(sceneverse_id)]
                                    random.shuffle(text_list)
                                    text_input = []
                                    for text in text_list[:10]:
                                        text_input.append(text[1])
                                    with torch.no_grad():
                                        text_ids = policy_net.rgb_encoder.tokenize(text_input).to(policy_net.rgb_encoder.device)
                                        text_fts, sep_fts = policy_net.rgb_encoder.model.encode_all_text(text_ids)

                                    sep_fts = sep_fts / torch.linalg.norm(sep_fts, dim=-1, keepdim=True)
                                    sampled_predicted_panorama_for_text_align.append(sep_fts @ predicted_panorama_fts[batch_id].T)
                                    sampled_gt_panorama_for_text_align.append(torch.tensor([shuffled_view_id]*sep_fts.shape[0], device=self.device))

                                    if fine_grained_sample_num < 1:
                                        # Fine-grained contrastive learning
                                        sampled_predicted_panorama_for_fine_grained_text_align.append(predicted_panorama_fts[batch_id:batch_id+1])
                                        sampled_gt_panorama_for_fine_grained_text_align.append(text_fts[0:1])
                                        fine_grained_sample_num += 1
                    

                for sampled_id in range(sampled_bev_num):

                    batch_selected_heading_angle = []
                    batch_selected_position = []
                    batch_selected_view_rgb = []
                    #batch_selected_view_depth = []
                    for b in range(batch_size):                
                        selected_nodes = random.choices(list(range(len(batch_angles[b]))), k=1)
                        node_id = selected_nodes[0]

                        selected_position = self.envs.call_at(b, "get_cand_real_pos", {"angle": batch_angles[b][node_id], "forward": batch_distances[b][node_id]})
                        selected_heading_angle = random.uniform(-math.pi,math.pi)
                        q1 = math.cos(selected_heading_angle/2)
                        q2 = math.sin(selected_heading_angle/2)
                        selected_rotation = np.quaternion(q1,0,q2,0)

                        camera_obs = self.envs.call_at(b, "get_observation",{"source_position":selected_position,"source_rotation":selected_rotation})

                        view_rgb = torch.tensor(camera_obs['rgb']).to(self.device).unsqueeze(0)
                        #view_depth = torch.tensor(camera_obs['depth']).to(self.device).unsqueeze(0)

                        batch_selected_heading_angle.append(selected_heading_angle)
                        batch_selected_position.append(selected_position)
                        batch_selected_view_rgb.append(view_rgb)
                        #batch_selected_view_depth.append(view_depth)

                    # Predict the bev features and the instance id
                    predicted_bev_fts, predicted_bev_feature_map, gt_label = policy_net.feature_fields.run_bev_encode(batch_selected_position, batch_selected_heading_angle,visualization=False)   
                    for batch_id in range(batch_size):
                        if gt_label[batch_id] != None:
                            #instance_id_to_label = batch_instance_id_to_label[batch_id]
                            selected_mask = (gt_label[batch_id]>0).view(-1)
                            selected_gt_label = gt_label[batch_id][selected_mask]
                            if selected_gt_label.shape[0] == 0:
                                continue

                            selected_bev_fts = predicted_bev_feature_map[batch_id][selected_mask]
                            for ray_id in range(selected_gt_label.shape[0]):
                                instance_id = int(selected_gt_label[ray_id].item())
                                category_id =  batch_instance_id_to_object_type[batch_id][instance_id][1] #instance_id_to_label[instance_id]
                                if category_id in self.category_dict:
                                    category_embedding_index = self.category_dict[category_id]
                                    sampled_gt_bev.append(category_embedding_index)
                                    sampled_predicted_bev_fts.append(selected_bev_fts[ray_id:ray_id+1])


                        # Get complex language supervision of bev grid map
                            # Upsample grid feature map to the size of bev rendering map
                            grid_map_height = policy_net.feature_fields.args.grid_map_height
                            grid_map_width = policy_net.feature_fields.args.grid_map_width
                            bev_height = policy_net.feature_fields.args.localization_map_height
                            bev_width = policy_net.feature_fields.args.localization_map_width
                            repeated_predicted_bev_fts = predicted_bev_fts[batch_id].view(grid_map_height,grid_map_width,-1).permute(2,0,1).unsqueeze(0)
                            repeated_predicted_bev_fts = nn.functional.interpolate(repeated_predicted_bev_fts, size=(bev_height,bev_width),mode='nearest')
                            repeated_predicted_bev_fts = repeated_predicted_bev_fts.squeeze(0).permute(1,2,0).view(bev_height*bev_width,-1)
                            selected_bev_fts = repeated_predicted_bev_fts[selected_mask]

                            window_size = 5
                            padding_size = window_size//2
                            fine_grained_bev_fts = F.pad(predicted_bev_fts[batch_id].view(grid_map_height,grid_map_width,-1).permute(2,0,1).unsqueeze(0),(padding_size,padding_size,padding_size,padding_size), "constant", 0).squeeze(0).permute(1,2,0)


                            selected_text = []
                            selected_instance_list = list(set(selected_gt_label.squeeze(1).cpu().numpy().tolist()))
                            random.shuffle(selected_instance_list)
                            scene_id = batch_scene_id[batch_id]
                            num_of_localization_text = 0
                            for selected_instance in selected_instance_list:
                                sceneverse_id = batch_instance_id_to_object_type[batch_id][selected_instance][0] # Convert HM3D_id to Sceneverse_id
                                if str(sceneverse_id) in self.hm3d_language_annotations[scene_id]:
                                    text_list = self.hm3d_language_annotations[scene_id][str(sceneverse_id)]
                                    random.shuffle(text_list)
                                    text_input = []
                                    for text in text_list[:1]:
                                        text_input.append(text[1])

                                    area_fts = selected_bev_fts[selected_gt_label.squeeze(1) == selected_instance]
                                    area_fts = area_fts.mean(dim=0, keepdim=True)
                                    with torch.no_grad():
                                        text_ids = policy_net.rgb_encoder.tokenize(text_input).to(policy_net.rgb_encoder.device)
                                        text_fts, sep_fts = policy_net.rgb_encoder.model.encode_all_text(text_ids)

                                    sampled_predicted_bev_for_text_align.append(area_fts.repeat(sep_fts.shape[0],1)) # Store area features with the number as same as text features
                                    sampled_gt_bev_for_text_align.append(sep_fts)

                                    fine_grained_region_mask = torch.zeros(gt_label[batch_id].shape,dtype=torch.float32,device=text_fts.device)
                                    fine_grained_region_mask[gt_label[batch_id]==selected_instance] = 1
                                    kernel_size = policy_net.feature_fields.args.localization_map_height // policy_net.feature_fields.args.grid_map_height
                                    fine_grained_region_mask = F.avg_pool2d(fine_grained_region_mask.view(1,1,policy_net.feature_fields.args.localization_map_height,policy_net.feature_fields.args.localization_map_width),kernel_size=kernel_size,stride=kernel_size).view(-1)

                                    fine_grained_region_id = torch.argmax(fine_grained_region_mask)
                                    fine_grained_region_x = fine_grained_region_id // policy_net.feature_fields.args.grid_map_height
                                    fine_grained_region_y = fine_grained_region_id % policy_net.feature_fields.args.grid_map_width
                                    fine_grained_region_x += padding_size
                                    fine_grained_region_y += padding_size

                                    fine_grained_region_fts = fine_grained_bev_fts[fine_grained_region_x-padding_size:fine_grained_region_x+padding_size+1,fine_grained_region_y-padding_size:fine_grained_region_y+padding_size+1].reshape(1,window_size*window_size,-1)

                                    sampled_predicted_bev_for_fine_grained_text_align.append(fine_grained_region_fts)
                                    sampled_gt_bev_for_fine_grained_text_align.append(text_fts[0:1])

                        '''
                        # Get object supervision of bev grid map, and target of localization map
                            selected_text = []
                            selected_instance_list = list(set(selected_gt_label.squeeze(1).cpu().numpy().tolist()))
                            selected_instance = random.choice(selected_instance_list)
                            num_of_localization_text = 0                               

                            category_id =  batch_instance_id_to_object_type[batch_id][selected_instance][1]
                            text_input = "The "+category_id   
                            print(text_input)                       
                            if num_of_localization_text < 1 and category_id != "wall" and category_id != "ceiling" and category_id != "floor" and category_id != "unknown": # Remove these so many but useless objects
                                # Text and target area for localization map
                                selected_text.append(text_input) # Choice a text for localization
                                gt_localization_map = torch.zeros(gt_label[batch_id].shape).to(selected_gt_label.device)
                                gt_localization_map[gt_label[batch_id] == selected_instance] = 1. # Target area is 1.
                                gt_localization_map = F.max_pool2d(gt_localization_map.view(1,1,policy_net.feature_fields.args.localization_map_height,policy_net.feature_fields.args.localization_map_width),kernel_size=3,stride=1,padding=1)
                                matplotlib.use('TkAgg')
                                plt.imshow(gt_localization_map.view(168,168).cpu().numpy())
                                plt.tight_layout()
                                plt.show()
                                gt_localization_map = gt_localization_map.view(-1)
                                sampled_gt_localization_map.append(gt_localization_map)

                                num_of_localization_text += 1


                            if len(selected_text) > 0:
                                # Predict the localization map
                                with torch.no_grad():
                                    text_ids = policy_net.rgb_encoder.tokenize(selected_text).to(policy_net.rgb_encoder.device)
                                    text_fts = policy_net.rgb_encoder.model.encode_all_text(text_ids)
                                    category_embedding_index = self.category_dict[category_id]
                                    ft_1 = predicted_bev_feature_map / torch.linalg.norm(predicted_bev_feature_map, dim=-1, keepdim=True)
                                    ft_2 = self.category_embeddings.to(self.device)
                                    sim_score = F.softmax(ft_1 @ ft_2.T * 100., dim=-1)[...,category_embedding_index]
                                    matplotlib.use('TkAgg')
                                    plt.imshow(sim_score.view(168,168).detach().cpu().numpy())
                                    plt.tight_layout()
                                    plt.show()                       
                        '''

                # Contrastive learning for bev map
                if len(sampled_predicted_bev_fts) > 0:
                    sampled_predicted_bev_fts = torch.cat(sampled_predicted_bev_fts,dim=0).to(torch.float32)
                    #sampled_predicted_bev_fts = sampled_predicted_bev_fts / torch.linalg.norm(sampled_predicted_bev_fts, dim=-1, keepdim=True)                  
                    logits = sampled_predicted_bev_fts @ self.category_embeddings.T * 100.
                    target = torch.tensor(sampled_gt_bev, device=self.device)
                    loss_5 += focal_loss(logits,target) / 5.
                    

                # Align the cosine similarity of bev map
                if len(sampled_predicted_bev_for_text_align) > 0:
                    sampled_predicted_bev_for_text_align = torch.cat(sampled_predicted_bev_for_text_align,dim=0).to(torch.float32)
                    sampled_predicted_bev_for_text_align = sampled_predicted_bev_for_text_align / torch.linalg.norm(sampled_predicted_bev_for_text_align, dim=-1, keepdim=True)
                    sampled_gt_bev_for_text_align = torch.cat(sampled_gt_bev_for_text_align,dim=0).to(torch.float32)
                    sampled_gt_bev_for_text_align = sampled_gt_bev_for_text_align / torch.linalg.norm(sampled_gt_bev_for_text_align, dim=-1, keepdim=True)
                    loss_6 += self.contrastive_loss(sampled_predicted_bev_for_text_align, sampled_gt_bev_for_text_align, logit_scale=100.) / 5.

                    sampled_predicted_bev_for_fine_grained_text_align = torch.cat(sampled_predicted_bev_for_fine_grained_text_align,dim=0).to(torch.float32)
                    sampled_gt_bev_for_fine_grained_text_align = torch.cat(sampled_gt_bev_for_fine_grained_text_align,dim=0).to(torch.float32)
                    loss_6 += self.fine_grained_contrastive_loss(sampled_predicted_bev_for_fine_grained_text_align,sampled_gt_bev_for_fine_grained_text_align) / 5.


            # Get next waypoint to move
            teacher_actions = self._teacher_action(batch_angles, batch_distances)
            env_actions = []
            for i in range(batch_size):
                if teacher_actions[i] == -1 or stepk == self.max_len-1:
                    env_actions.append({'action':
                        {'action': 0, 'action_args':{}}})
                else:
                    env_actions.append({'action':
                        {'action': 4,  # HIGHTOLOW
                        'action_args':{
                            'angle': batch_angles[i][teacher_actions[i].item()], 
                            'distance': batch_distances[i][teacher_actions[i].item()],
                        }}})

            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in
                                                zip(*outputs)]
            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(len(dones)))):
                    if dones[i]:
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        self.envs.pause_at(i)
                        observations.pop(i)
                        policy_net.feature_fields.pop(i) # Very important, for some navigation processes finished in habitat simulator
                        batch_scene_id.pop(i)
                        batch_pcd_xyz.pop(i)
                        batch_pcd_label.pop(i)
                        #batch_instance_id_to_label.pop(i)
                        batch_instance_id_to_object_type.pop(i)
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        # Contrastive learning for view and panorama
        loss_5, loss_6 = loss_5/total_actions, loss_6/total_actions

        sampled_predicted_view_fts = torch.cat(sampled_predicted_view_fts,dim=0).to(torch.float32)
        sampled_predicted_view_fts = sampled_predicted_view_fts / torch.linalg.norm(sampled_predicted_view_fts, dim=-1, keepdim=True)
        sampled_gt_view_fts = torch.cat(sampled_gt_view_fts,dim=0).to(torch.float32)
        sampled_gt_view_fts = sampled_gt_view_fts / torch.linalg.norm(sampled_gt_view_fts, dim=-1, keepdim=True)
        loss_1 += self.contrastive_loss(sampled_predicted_view_fts, sampled_gt_view_fts)
        loss_1 += (1. - (sampled_predicted_view_fts * sampled_gt_view_fts).sum(-1)).mean()

        sampled_predicted_panorama_fts = torch.cat(sampled_predicted_panorama_fts,dim=0).to(torch.float32)
        sampled_predicted_panorama_fts = sampled_predicted_panorama_fts / torch.linalg.norm(sampled_predicted_panorama_fts, dim=-1, keepdim=True)
        sampled_gt_panorama_fts = torch.cat(sampled_gt_panorama_fts,dim=0).to(torch.float32)
        sampled_gt_panorama_fts = sampled_gt_panorama_fts / torch.linalg.norm(sampled_gt_panorama_fts, dim=-1, keepdim=True)
        loss_2 += self.contrastive_loss(sampled_predicted_panorama_fts, sampled_gt_panorama_fts)
        loss_2 += (1. - (sampled_predicted_panorama_fts * sampled_gt_panorama_fts).sum(-1)).mean()


        # Contrastive learning for region features
        if len(sampled_predicted_region_fts) > 0:
            sampled_predicted_region_fts = torch.cat(sampled_predicted_region_fts,dim=0).to(torch.float32)
            sampled_predicted_region_fts = sampled_predicted_region_fts / torch.linalg.norm(sampled_predicted_region_fts, dim=-1, keepdim=True)
            logits = sampled_predicted_region_fts @ self.category_embeddings.T * 100.
            target = torch.tensor(sampled_gt_region, device=self.device)
            loss_3 += focal_loss(logits,target) / 5.

        # Align the cosine similarity of panorama features
        if len(sampled_predicted_panorama_for_text_align) > 0:
            sampled_predicted_panorama_for_text_align = torch.cat(sampled_predicted_panorama_for_text_align,dim=0).to(torch.float32) * 100.
            sampled_gt_panorama_for_text_align = torch.cat(sampled_gt_panorama_for_text_align,dim=0)
            loss_4 += F.cross_entropy(sampled_predicted_panorama_for_text_align, sampled_gt_panorama_for_text_align)

            sampled_predicted_panorama_for_fine_grained_text_align = torch.cat(sampled_predicted_panorama_for_fine_grained_text_align,dim=0).to(torch.float32)
            sampled_gt_panorama_for_fine_grained_text_align = torch.cat(sampled_gt_panorama_for_fine_grained_text_align,dim=0).to(torch.float32)
            loss_4 += self.fine_grained_contrastive_loss(sampled_predicted_panorama_for_fine_grained_text_align,sampled_gt_panorama_for_fine_grained_text_align) / 5.
            
        print("HM3D","loss_1:"+str(loss_1)[:13],"loss_2:"+str(loss_2)[:13], "loss_3:"+str(loss_3)[:13],"loss_4:"+str(loss_4)[:13])

        print("HM3D","loss_5:"+str(loss_5)[:13],"loss_6:"+str(loss_6)[:13])

        loss += loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6

        policy_net.feature_fields.delete_feature_fields()

        return loss


    def run_on_structured3d(self, mode):
        loss = 0.
        loss_1 = loss_2 = loss_3 = loss_4 = loss_5 = loss_6 =  0.
        policy_net = self.policy.net
        if hasattr(self.policy.net, 'module'):
            policy_net = self.policy.net.module

        batch_size = self.batch_size * 4 # Larger batch size is better, so different from batch_size of hm3d is also ok

        camera_info = np.loadtxt('data/Structured3D/scene_00000/2D_rendering/485142/perspective/full/0/camera_pose.txt')
        view_height = policy_net.feature_fields.args.view_height
        view_width = policy_net.feature_fields.args.view_width
        camera_rot, camera_trans, init_camera_intrinsic = self.parse_camera_info(camera_info, view_height, view_width)
                
        batch_camera_intrinsic = []
        batch_grid_ft = []
        batch_depth = []
        batch_rot = []
        batch_trans = []
        batch_target_rot, batch_target_trans, batch_rgb_image = [], [], []
        batch_view_region_image, batch_view_region_id = [], []
        batch_bev_region_image, batch_bev_region_info = [], []

        # Get the instance label
        batch_pcd_xyz = []
        batch_pcd_label = []
        batch_instance_id_to_label = []
        batch_scene_id = []
        for batch_id in range(batch_size):
            scene_id = random.choice(list(self.structure_3d_scenes.keys()))
            file_path = self.structure_3d_scenes[scene_id]
            '''
            if task_id == 0:
                scene_id = random.choice(list(self.structure_3d_scenes.keys()))
                file_path = self.structure_3d_scenes[scene_id]
            else:
                scene_id = random.choice(list(set(self.Structured3D_pcd_with_global_alignment.keys()) & set(self.Structured3D_language_annotations.keys())))
                file_path = self.structure_3d_scenes[scene_id]
            '''
            batch_scene_id.append(scene_id)

            '''
            # Get the instance label
            pcd_xyz = []
            pcd_label = []
            try:
                if scene_id in self.Structured3D_pcd_with_global_alignment:
                    
                    pcd_file_list = self.Structured3D_pcd_with_global_alignment[scene_id]
                    for pcd_file in pcd_file_list:
                        pcd_file = torch.load(pcd_file)
                        pcd_xyz.append(torch.tensor(pcd_file[0]))
                        pcd_label.append(torch.tensor(pcd_file[2]))
                    batch_pcd_xyz.append(torch.cat(pcd_xyz,0))
                    batch_pcd_label.append(torch.cat(pcd_label,0))

                    instance_id_to_label_list = self.Structured3D_instance_id_to_label[scene_id]
                    label_dict = {}
                    for label_file in instance_id_to_label_list:
                        label_file = torch.load(label_file)
                        label_dict.update(label_file)
                    batch_instance_id_to_label.append(label_dict)
                else:
                    batch_pcd_xyz.append(None)
                    batch_pcd_label.append(None)
                    batch_instance_id_to_label.append(None)
            except:
                batch_pcd_xyz.append(None)
                batch_pcd_label.append(None)
                batch_instance_id_to_label.append(None)
                print("File of Structured3D_pcd_with_global_alignment or Structured3D_instance_id_to_label has error, skip...")
            '''

            room_list = [file_path+'/'+item+'/perspective/full' for item in os.listdir(file_path)]
            image_list = []
            for i in room_list:
                if os.path.exists(i):
                    for j in os.listdir(i):
                        image_list.append(i+'/'+j)
            if len(image_list) == 0:
                print("Miss",file_path,"!!!!!!!!!!!!")

            target_image = random.choice(image_list)
            batch_bev_region_info.append(random.choice(image_list))
            intrinsic_list = []
            R_list = []
            T_list = []
            rgb_list = []
            depth_list = []
            
            for image_id in image_list:
                camera_info = np.loadtxt(os.path.join(image_id, 'camera_pose.txt'))
                rot, trans, intrinsic =self.parse_camera_info(camera_info,720, 1280)
                intrinsic_list.append(intrinsic)
                extrinsic = np.eye(4)
                extrinsic[:3,:3] = rot
                extrinsic = np.linalg.inv(extrinsic)
                R = extrinsic[:3,:3]
                T = trans.reshape(3,1)
                rgb_image = Image.open(image_id + '/rgb_rawlight.png').convert('RGB')
                depth_image = Image.open(image_id + '/depth.png')

                if image_id == batch_bev_region_info[batch_id]:

                    crop_shape = (336,448) # !!!!!!!!!!!
                    nh = random.randint(0, rgb_image.size[1] - crop_shape[0]) # Note that swap h,w for PIL image
                    nw = random.randint(0, rgb_image.size[0] - crop_shape[1])
                    image_crop = rgb_image.crop((nh, nw, nh + crop_shape[0], nw + crop_shape[1]))
                    batch_bev_region_image.append(torch.tensor(np.asarray(image_crop)).unsqueeze(0))
                    batch_bev_region_info[batch_id] = [R,T,intrinsic,torch.tensor(np.asarray(depth_image)),(nh, nw, nh + crop_shape[0], nw + crop_shape[1])]

                if image_id == target_image:
                    batch_target_rot.append(R)
                    batch_target_trans.append(T)
                    batch_rgb_image.append(torch.tensor(np.asarray(rgb_image)).unsqueeze(0))

                    # crop a subregion image for train
                    feature_map_height = policy_net.feature_fields.args.view_height
                    feature_map_width = policy_net.feature_fields.args.view_width
                    
                    crop_shape = (336,448) # !!!!!!!!!!!
                    nh = random.randint(0, rgb_image.size[1] - crop_shape[0]) # Note that swap h,w for PIL image
                    nw = random.randint(0, rgb_image.size[0] - crop_shape[1])
                    image_crop = rgb_image.crop((nh, nw, nh + crop_shape[0], nw + crop_shape[1]))
                    batch_view_region_image.append(torch.tensor(np.asarray(image_crop)).unsqueeze(0))
                    region_x = int((nh + nh + crop_shape[0])/2./rgb_image.size[1]*(feature_map_height-1))
                    region_y = int((nw + nw + crop_shape[1])/2./rgb_image.size[0]*(feature_map_width-1))
                    batch_view_region_id.append((region_x,region_y))
                    

                    if len(image_list) > 1: # Some scenes only have one image
                        continue

                R_list.append(R)
                T_list.append(T)
                rgb_list.append(torch.tensor(np.asarray(rgb_image)).unsqueeze(0))
                depth_list.append(torch.tensor(np.asarray(depth_image)).unsqueeze(0))

            if len(rgb_list)==0:
                print("Miss", image_list,"!!!!!!!!!!!!")

            rgb_list = torch.cat(rgb_list,dim=0)
            depth_list = torch.cat(depth_list,dim=0)
            with torch.no_grad():
                _, grid_fts = policy_net.rgb_encoder({'rgb':rgb_list})

            batch_camera_intrinsic.append(intrinsic_list)
            batch_rot.append(R_list)
            batch_trans.append(T_list)
            batch_grid_ft.append(grid_fts.cpu().numpy())
            batch_depth.append(depth_list)

        
        policy_net.feature_fields.reset(batch_size=batch_size, mode='structured3d',camera_intrinsic=init_camera_intrinsic)#, batch_gt_pcd_xyz=batch_pcd_xyz, batch_gt_pcd_label=batch_pcd_label)  # Reset the settings of 3D feature fields
        policy_net.feature_fields.update_feature_fields(batch_depth, batch_grid_ft, batch_camera_intrinsic, batch_rot, batch_trans, depth_scale=1000., depth_trunc=1000.)

       
        sampled_predicted_view_fts = []
        sampled_gt_view_fts = []
        sampled_predicted_panorama_fts = []
        sampled_gt_panorama_fts = []

        sampled_predicted_region_fts = []
        sampled_gt_region = []
        sampled_predicted_panorama_for_text_align = []       
        sampled_gt_panorama_for_text_align = []

        sampled_predicted_bev = []
        sampled_gt_bev = []
        sampled_predicted_bev_for_text_align = []
        sampled_gt_bev_for_text_align = []

        # Get ground truth novel view CLIP features
        with torch.no_grad():
            gt_view_fts, _ = policy_net.rgb_encoder({'rgb':torch.cat(batch_rgb_image,dim=0)}) 
                
            # visual region alignment
            gt_region_fts, _ = policy_net.rgb_encoder({'rgb':torch.cat(batch_view_region_image,dim=0).to(self.device)}) 

        sampled_gt_view_fts = [gt_view_fts]
        sampled_gt_region = [gt_region_fts]

        # Predict the novel view features and the instance id within this view
        predicted_view_fts, predicted_region_fts, predicted_feature_map, gt_label = policy_net.feature_fields.run_view_encode(batch_rot=batch_target_rot, batch_trans=batch_target_trans,visualization=False)
        sampled_predicted_view_fts = [predicted_view_fts]

        # Get visual supervision of region features
        region_feature_map = predicted_region_fts.view(batch_size,policy_net.feature_fields.args.view_height,policy_net.feature_fields.args.view_width,-1)
        for batch_id in range(batch_size):
            patch_x,patch_y = batch_view_region_id[batch_id]
            selected_region_fts = region_feature_map[batch_id][patch_x,patch_y].unsqueeze(0)
            sampled_predicted_region_fts.append(selected_region_fts)


        # Predict the panorama features and the instance id within this panorama
        predicted_panorama_fts, shuffled_view_id = policy_net.feature_fields.run_panorama_encode(batch_rot=batch_target_rot, batch_trans=batch_target_trans,view_shuffle=True,visualization=False)
        sampled_predicted_panorama_fts = [predicted_panorama_fts[:,shuffled_view_id]] # Only store the shuffled view (forward-facing)
        sampled_gt_panorama_fts = [gt_view_fts]


        # visual bev alignment
        with torch.no_grad():
            gt_bev_fts, _ = policy_net.rgb_encoder({'rgb':torch.cat(batch_bev_region_image,dim=0).to(self.device)}) 

        # Predict the bev features and the instance id
        predicted_bev_fts, predicted_bev_feature_map, gt_label = policy_net.feature_fields.run_bev_encode(batch_rot=batch_target_rot, batch_trans=batch_target_trans,visualization=False)

        # Get visual supervision of bev features
        bev_target_depth = [batch_bev_region_info[batch_id][3] for batch_id in range(batch_size)]
        bev_target_bbox = [batch_bev_region_info[batch_id][4] for batch_id in range(batch_size)]
        bev_target_R = [batch_bev_region_info[batch_id][0] for batch_id in range(batch_size)]
        bev_target_T = [batch_bev_region_info[batch_id][1] for batch_id in range(batch_size)]
        bev_target_intrinsic = [batch_bev_region_info[batch_id][2] for batch_id in range(batch_size)]

        batch_bev_gt_label = policy_net.feature_fields.get_bev_visual_target(bev_target_depth, bev_target_bbox, bev_target_intrinsic, bev_target_R, bev_target_T, batch_target_rot, batch_target_trans, depth_scale=1000.0, depth_trunc=1000.0)

        predicted_bev_fts = predicted_bev_fts / torch.linalg.norm(predicted_bev_fts, dim=-1, keepdim=True)
        gt_bev_fts = gt_bev_fts / torch.linalg.norm(gt_bev_fts, dim=-1, keepdim=True)
        for batch_id in range(batch_size):
            sampled_predicted_bev.append( gt_bev_fts[batch_id:batch_id+1] @ predicted_bev_fts[batch_id].T )
            sampled_gt_bev.append(torch.tensor([batch_bev_gt_label[batch_id]],device=self.device))


        # Contrastive learning for view and panorama
        sampled_predicted_view_fts = torch.cat(sampled_predicted_view_fts,dim=0).to(torch.float32)
        sampled_predicted_view_fts = sampled_predicted_view_fts / torch.linalg.norm(sampled_predicted_view_fts, dim=-1, keepdim=True)
        sampled_gt_view_fts = torch.cat(sampled_gt_view_fts,dim=0).to(torch.float32)
        sampled_gt_view_fts = sampled_gt_view_fts / torch.linalg.norm(sampled_gt_view_fts, dim=-1, keepdim=True)
        loss_1 += self.contrastive_loss(sampled_predicted_view_fts, sampled_gt_view_fts)
        loss_1 += (1. - (sampled_predicted_view_fts * sampled_gt_view_fts).sum(-1)).mean()

        sampled_predicted_panorama_fts = torch.cat(sampled_predicted_panorama_fts,dim=0).to(torch.float32)
        sampled_predicted_panorama_fts = sampled_predicted_panorama_fts / torch.linalg.norm(sampled_predicted_panorama_fts, dim=-1, keepdim=True)
        sampled_gt_panorama_fts = torch.cat(sampled_gt_panorama_fts,dim=0).to(torch.float32)
        sampled_gt_panorama_fts = sampled_gt_panorama_fts / torch.linalg.norm(sampled_gt_panorama_fts, dim=-1, keepdim=True)
        loss_2 += self.contrastive_loss(sampled_predicted_panorama_fts, sampled_gt_panorama_fts)
        loss_2 += (1. - (sampled_predicted_panorama_fts * sampled_gt_panorama_fts).sum(-1)).mean()

        # Contrastive learning for region features
        if len(sampled_predicted_region_fts) > 0:
            sampled_predicted_region_fts = torch.cat(sampled_predicted_region_fts,dim=0).to(torch.float32)
            sampled_predicted_region_fts = sampled_predicted_region_fts / torch.linalg.norm(sampled_predicted_region_fts, dim=-1, keepdim=True)
            sampled_gt_region = torch.cat(sampled_gt_region,dim=0).to(torch.float32)
            sampled_gt_region = sampled_gt_region / torch.linalg.norm(sampled_gt_region, dim=-1, keepdim=True)
            loss_3 += self.contrastive_loss(sampled_predicted_region_fts, sampled_gt_region) 
            loss_3 += (1. - (sampled_predicted_region_fts * sampled_gt_region).sum(-1)).mean()

        # Align the cosine similarity of panorama features
        if len(sampled_predicted_panorama_for_text_align) > 0:
            sampled_predicted_panorama_for_text_align = torch.cat(sampled_predicted_panorama_for_text_align,dim=0).to(torch.float32)
            sampled_predicted_panorama_for_text_align = sampled_predicted_panorama_for_text_align / torch.linalg.norm(sampled_predicted_panorama_for_text_align, dim=-1, keepdim=True)
            sampled_gt_panorama_for_text_align = torch.cat(sampled_gt_panorama_for_text_align,dim=0).to(torch.float32)
            sampled_gt_panorama_for_text_align = sampled_gt_panorama_for_text_align / torch.linalg.norm(sampled_gt_panorama_for_text_align, dim=-1, keepdim=True)
            loss_4 += self.contrastive_loss(sampled_predicted_panorama_for_text_align, sampled_gt_panorama_for_text_align, logit_scale=100.) 

        print("Structured3D","loss_1:"+str(loss_1)[:13],"loss_2:"+str(loss_2)[:13], "loss_3:"+str(loss_3)[:13],"loss_4:"+str(loss_4)[:13])


        # Contrastive learning for bev map
        if len(sampled_predicted_bev) > 0:
            sampled_predicted_bev = torch.cat(sampled_predicted_bev,dim=0).to(torch.float32) * 10.
            sampled_gt_bev = torch.cat(sampled_gt_bev,dim=0)
            loss_5 += F.cross_entropy(sampled_predicted_bev, sampled_gt_bev) / 3.

        # Align the cosine similarity of bev map
        if len(sampled_predicted_bev_for_text_align) > 0:
            sampled_predicted_bev_for_text_align = torch.cat(sampled_predicted_bev_for_text_align,dim=0).to(torch.float32)
            sampled_predicted_bev_for_text_align = sampled_predicted_bev_for_text_align / torch.linalg.norm(sampled_predicted_bev_for_text_align, dim=-1, keepdim=True)
            sampled_gt_bev_fts_for_text_align = torch.cat(sampled_gt_bev_fts_for_text_align,dim=0).to(torch.float32)
            sampled_gt_bev_fts_for_text_align = sampled_gt_bev_fts_for_text_align / torch.linalg.norm(sampled_gt_bev_fts_for_text_align, dim=-1, keepdim=True)
            loss_6 += self.contrastive_loss(sampled_predicted_bev_for_text_align, sampled_gt_bev_fts_for_text_align, logit_scale=100.) / 10.


        print("Structured3D","loss_5:"+str(loss_5)[:13],"loss_6:"+str(loss_6)[:13])

        loss += loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6
        policy_net.feature_fields.delete_feature_fields()
        return loss


    def run_on_scannet(self, mode):
        loss = 0.
        loss_1 = loss_2 = loss_3 = loss_4 = loss_5 = loss_6 =  0.
        policy_net = self.policy.net
        if hasattr(self.policy.net, 'module'):
            policy_net = self.policy.net.module

        batch_size = self.batch_size * 8 # Larger batch size is better, so different from batch_size of hm3d is also ok

        num_of_sampled_images = 30 # Number of sampled images to construct the feature fields
        init_camera_intrinsic = np.eye(4)
        with open('data/ScanNet/scannet_train_images/frames_square/scene0000_00/intrinsic_depth.txt', 'r') as file:  
            numbers = [line.strip() for line in file]
        for i in range(4):  
            for j in range(4): 
                init_camera_intrinsic[i][j] = float(numbers[i].split()[j])
        init_camera_intrinsic[0][0] *= (policy_net.feature_fields.args.view_width / 320) / 2.
        init_camera_intrinsic[1][1] *= (policy_net.feature_fields.args.view_height / 240) / 2.
            
        batch_camera_intrinsic = []
        batch_grid_ft = []
        batch_depth = []
        batch_rot = []
        batch_trans = []
        batch_target_rot, batch_target_trans, batch_rgb_image = [], [], []

        # Get the instance label
        batch_pcd_xyz = []
        batch_pcd_label = []
        batch_instance_id_to_label = []
        batch_scene_id = []
        for batch_id in range(batch_size):

            scene_id = random.choice(list(set(self.ScanNet_pcd_with_global_alignment.keys()) & set(self.ScanNet_language_annotations.keys())))
            file_path = self.scannet_3d_scenes[scene_id]
            batch_scene_id.append(scene_id)

            # Get the instance label
            pcd_xyz = []
            pcd_label = []
            if scene_id in self.ScanNet_pcd_with_global_alignment:
                    
                pcd_file_list = self.ScanNet_pcd_with_global_alignment[scene_id]
                for pcd_file in pcd_file_list:
                    pcd_file = torch.load(pcd_file)
                    align_matrix = torch.tensor(np.linalg.inv(self.scannet_align_matrix[scene_id])).to(self.device).to(torch.float32) # Align the coordinate using align_matrix
                    pts = np.ones((pcd_file[0].shape[0], 4), dtype=np.float32)
                    pts[:, 0:3] = pcd_file[0]
                    aligned_xyz = (torch.tensor(pts).to(self.device) @ align_matrix.T)[:, :3].cpu()
                    pcd_xyz.append(aligned_xyz)
                    pcd_label.append(torch.tensor(pcd_file[3])) # 3 not 2, different from hm3d and structered3d
                batch_pcd_xyz.append(torch.cat(pcd_xyz,0))
                batch_pcd_label.append(torch.cat(pcd_label,0))

                instance_id_to_label_list = self.ScanNet_instance_id_to_label[scene_id]
                label_dict = {}
                for label_file in instance_id_to_label_list:
                    label_file = torch.load(label_file)
                    label_dict.update(label_file)
                batch_instance_id_to_label.append(label_dict)
            else:
                batch_pcd_xyz.append(None)
                batch_pcd_label.append(None)
                batch_instance_id_to_label.append(None)

            image_list = []
            for image_id in range(1000):
                image_id = image_id * 20   
                image_path = file_path + '/color/' + str(image_id) + ".jpg"
                if not os.path.exists(image_path):
                    break
                image_list.append(str(image_id))

            random.shuffle(image_list)
            image_list = image_list[:num_of_sampled_images]
            target_image = random.choice(image_list)
            intrinsic_list = []
            R_list = []
            T_list = []
            rgb_list = []
            depth_list = []
            for image_id in image_list:
                intrinsic = np.eye(4)
                with open(file_path + '/intrinsic_depth.txt', 'r') as file:  
                    intrinsic_raw = [line.strip() for line in file]
                for i in range(4):  
                    for j in range(4): 
                        intrinsic[i][j] = float(intrinsic_raw[i].split()[j])

                # divide 2 is necessary for camera intrinsics of scannet dataset
                intrinsic[0][0] =  intrinsic[0][0] / 2.
                intrinsic[1][1] =  intrinsic[1][1] / 2.
                intrinsic[0][2] =  intrinsic[0][2] / 2.
                intrinsic[1][2] =  intrinsic[1][2] / 2.
                intrinsic_list.append(intrinsic)
                extrinsic = np.eye(4)
                with open(file_path + '/pose/' + image_id + '.txt', 'r') as file:  
                    extrinsic_raw = [line.strip() for line in file]
                for i in range(4):  
                    for j in range(4): 
                        extrinsic[i][j] = float(extrinsic_raw[i].split()[j])
                R = extrinsic[:3,:3]
                T = extrinsic[:3,3:4]

                rgb_image = np.asarray(Image.open(file_path + '/color/' + image_id + ".jpg"))
                depth_image = np.asarray(Image.open(file_path + '/depth/' + image_id + ".png"))

                if image_id == target_image:
                    batch_target_rot.append(R)
                    batch_target_trans.append(T)
                    batch_rgb_image.append(torch.tensor(rgb_image).unsqueeze(0))
                    if len(image_list) > 1: # Some scenes only have one image
                        continue

                R_list.append(R)
                T_list.append(T)
                rgb_list.append(torch.tensor(rgb_image).unsqueeze(0))
                depth_list.append(torch.tensor(depth_image).unsqueeze(0))

            rgb_list = torch.cat(rgb_list,dim=0)
            depth_list = torch.cat(depth_list,dim=0)
            with torch.no_grad():
                _, grid_fts = policy_net.rgb_encoder({'rgb':rgb_list})

            batch_camera_intrinsic.append(intrinsic_list)
            batch_rot.append(R_list)
            batch_trans.append(T_list)
            batch_grid_ft.append(grid_fts.cpu().numpy())
            batch_depth.append(depth_list)

        policy_net.feature_fields.reset(batch_size=batch_size, mode='scannet',camera_intrinsic=init_camera_intrinsic, batch_gt_pcd_xyz=batch_pcd_xyz, batch_gt_pcd_label=batch_pcd_label)  # Reset the settings of 3D feature fields
        policy_net.feature_fields.update_feature_fields(batch_depth, batch_grid_ft, batch_camera_intrinsic, batch_rot, batch_trans, depth_scale=1000., depth_trunc=1000.)
        

        sampled_predicted_view_fts = []
        sampled_gt_view_fts = []
        sampled_predicted_panorama_fts = []
        sampled_gt_panorama_fts = []

        sampled_predicted_region_fts = []
        sampled_gt_region = []
        sampled_predicted_panorama_for_text_align = []       
        sampled_gt_panorama_for_text_align = []

        sampled_predicted_panorama_for_fine_grained_text_align = []       
        sampled_gt_panorama_for_fine_grained_text_align = []

        sampled_predicted_bev_fts = []
        sampled_gt_bev = []
        sampled_predicted_bev_for_text_align = []
        sampled_gt_bev_for_text_align = []

        sampled_predicted_bev_for_fine_grained_text_align = []       
        sampled_gt_bev_for_fine_grained_text_align = []

        # Get ground truth novel view CLIP features
        with torch.no_grad():
            gt_view_fts, _ = policy_net.rgb_encoder({'rgb':torch.cat(batch_rgb_image,dim=0)}) 
                
        sampled_gt_view_fts = [gt_view_fts]
        # Predict the novel view features and the instance id within this view
        predicted_view_fts, predicted_region_fts, predicted_feature_map, gt_label = policy_net.feature_fields.run_view_encode(batch_rot=batch_target_rot, batch_trans=batch_target_trans,visualization=False)
        sampled_predicted_view_fts = [predicted_view_fts]

        # Get language supervision of region features
        for batch_id in range(batch_size):
            if gt_label[batch_id] != None:
                instance_id_to_label = batch_instance_id_to_label[batch_id]
                selected_mask = (gt_label[batch_id]>=0).view(-1)
                selected_gt_label = gt_label[batch_id][selected_mask]
                if selected_gt_label.shape[0] == 0:
                    continue

                selected_region_fts = predicted_feature_map[batch_id][selected_mask]

                for ray_id in range(selected_gt_label.shape[0]):
                    instance_id = int(selected_gt_label[ray_id].item())                        
                    category_id = instance_id_to_label[instance_id]
                    if category_id in self.category_dict:
                        category_embedding_index = self.category_dict[category_id]
                        sampled_gt_region.append(category_embedding_index)
                        sampled_predicted_region_fts.append(selected_region_fts[ray_id:ray_id+1])
                    
                #import matplotlib.pyplot as plt
                #plt.figure("Image")
                #plt.imshow(rgb_image)
                #plt.title('image') 
                #plt.show()

        # Predict the panorama features and the instance id within this panorama
        predicted_panorama_fts, shuffled_view_id = policy_net.feature_fields.run_panorama_encode(batch_rot=batch_target_rot, batch_trans=batch_target_trans,view_shuffle=True,visualization=False)
        sampled_predicted_panorama_fts = [predicted_panorama_fts[:,shuffled_view_id]] # Only store the shuffled view (forward-facing)
        sampled_gt_panorama_fts = [gt_view_fts]

        predicted_panorama_fts = predicted_panorama_fts / torch.linalg.norm(predicted_panorama_fts, dim=-1, keepdim=True)

        # Get language supervision of panorama features
        for batch_id in range(batch_size):
            if gt_label[batch_id] != None:
                selected_mask = (gt_label[batch_id]>=0).view(-1)
                selected_gt_label = gt_label[batch_id][selected_mask]
                if selected_gt_label.shape[0] == 0:
                    continue
                selected_gt_label = list(set(selected_gt_label.squeeze(1).cpu().numpy().tolist()))
                random.shuffle(selected_gt_label)

                # Text for panorama
                fine_grained_sample_num = 0
                scene_id = batch_scene_id[batch_id]
                for selected_instance in selected_gt_label:
                    if scene_id in self.ScanNet_language_annotations and str(selected_instance) in self.ScanNet_language_annotations[scene_id]:
                        text_list = self.ScanNet_language_annotations[scene_id][str(selected_instance)]
                        random.shuffle(text_list)
                        text_input = []
                        for text in text_list[:10]:
                            text_input.append(text[1])
                        with torch.no_grad():
                            text_ids = policy_net.rgb_encoder.tokenize(text_input).to(policy_net.rgb_encoder.device)
                            text_fts, sep_fts = policy_net.rgb_encoder.model.encode_all_text(text_ids)

                        sep_fts = sep_fts / torch.linalg.norm(sep_fts, dim=-1, keepdim=True)
                        sampled_predicted_panorama_for_text_align.append(sep_fts @ predicted_panorama_fts[batch_id].T)
                        sampled_gt_panorama_for_text_align.append(torch.tensor([shuffled_view_id]*sep_fts.shape[0], device=self.device))

                        if fine_grained_sample_num < 1:
                            # Fine-grained contrastive learning
                            sampled_predicted_panorama_for_fine_grained_text_align.append(predicted_panorama_fts[batch_id:batch_id+1])
                            sampled_gt_panorama_for_fine_grained_text_align.append(text_fts[0:1])
                            fine_grained_sample_num += 1

        # Predict the bev features and the instance id
        predicted_bev_fts, predicted_bev_feature_map, gt_label = policy_net.feature_fields.run_bev_encode(batch_rot=batch_target_rot, batch_trans=batch_target_trans,visualization=False)               
        for batch_id in range(batch_size):
            if gt_label[batch_id] != None:
                instance_id_to_label = batch_instance_id_to_label[batch_id]
                selected_mask = (gt_label[batch_id]>=0).view(-1)
                selected_gt_label = gt_label[batch_id][selected_mask]
                if selected_gt_label.shape[0] == 0:
                    continue

                selected_bev_fts = predicted_bev_feature_map[batch_id][selected_mask]
                for ray_id in range(selected_gt_label.shape[0]):
                    instance_id = int(selected_gt_label[ray_id].item())
                    category_id = instance_id_to_label[instance_id]
                    if category_id in self.category_dict:
                        category_embedding_index = self.category_dict[category_id]
                        sampled_gt_bev.append(category_embedding_index)
                        sampled_predicted_bev_fts.append(selected_bev_fts[ray_id:ray_id+1])  


                # Get language supervision of bev grid map, and target of localization map
                # Upsample grid feature map to the size of bev rendering map
                grid_map_height = policy_net.feature_fields.args.grid_map_height
                grid_map_width = policy_net.feature_fields.args.grid_map_width
                bev_height = policy_net.feature_fields.args.localization_map_height
                bev_width = policy_net.feature_fields.args.localization_map_width
                repeated_predicted_bev_fts = predicted_bev_fts[batch_id].view(grid_map_height,grid_map_width,-1).permute(2,0,1).unsqueeze(0)
                repeated_predicted_bev_fts = nn.functional.interpolate(repeated_predicted_bev_fts, size=(bev_height,bev_width),mode='nearest')
                repeated_predicted_bev_fts = repeated_predicted_bev_fts.squeeze(0).permute(1,2,0).view(bev_height*bev_width,-1)
                selected_bev_fts = repeated_predicted_bev_fts[selected_mask]


                window_size = 5
                padding_size = window_size//2
                fine_grained_bev_fts = F.pad(predicted_bev_fts[batch_id].view(grid_map_height,grid_map_width,-1).permute(2,0,1).unsqueeze(0),(padding_size,padding_size,padding_size,padding_size), "constant", 0).squeeze(0).permute(1,2,0)

                selected_text = []
                selected_instance_list = list(set(selected_gt_label.squeeze(1).cpu().numpy().tolist()))
                random.shuffle(selected_instance_list)
                scene_id = batch_scene_id[batch_id]
                num_of_localization_text = 0
                for selected_instance in selected_instance_list:
                    if str(selected_instance) in self.ScanNet_language_annotations[scene_id]:
                        text_list = self.ScanNet_language_annotations[scene_id][str(selected_instance)]
                        random.shuffle(text_list)
                        text_input = []
                        for text in text_list[:1]:
                            text_input.append(text[1])

                        area_fts = selected_bev_fts[selected_gt_label.squeeze(1) == selected_instance]
                        area_fts = area_fts.mean(dim=0, keepdim=True)
                        with torch.no_grad():
                            text_ids = policy_net.rgb_encoder.tokenize(text_input).to(policy_net.rgb_encoder.device)
                            text_fts, sep_fts = policy_net.rgb_encoder.model.encode_all_text(text_ids)

                        sampled_predicted_bev_for_text_align.append(area_fts.repeat(sep_fts.shape[0],1)) # Store area features with the number as same as text features
                        sampled_gt_bev_for_text_align.append(sep_fts)

                        fine_grained_region_mask = torch.zeros(gt_label[batch_id].shape,dtype=torch.float32,device=text_fts.device)
                        fine_grained_region_mask[gt_label[batch_id]==selected_instance] = 1
                        kernel_size = policy_net.feature_fields.args.localization_map_height // policy_net.feature_fields.args.grid_map_height
                        fine_grained_region_mask = F.avg_pool2d(fine_grained_region_mask.view(1,1,policy_net.feature_fields.args.localization_map_height,policy_net.feature_fields.args.localization_map_width),kernel_size=kernel_size,stride=kernel_size).view(-1)

                        fine_grained_region_id = torch.argmax(fine_grained_region_mask)
                        fine_grained_region_x = fine_grained_region_id // policy_net.feature_fields.args.grid_map_height
                        fine_grained_region_y = fine_grained_region_id % policy_net.feature_fields.args.grid_map_width
                        fine_grained_region_x += padding_size
                        fine_grained_region_y += padding_size

                        fine_grained_region_fts = fine_grained_bev_fts[fine_grained_region_x-padding_size:fine_grained_region_x+padding_size+1,fine_grained_region_y-padding_size:fine_grained_region_y+padding_size+1].reshape(1,window_size*window_size,-1)

                        sampled_predicted_bev_for_fine_grained_text_align.append(fine_grained_region_fts)
                        sampled_gt_bev_for_fine_grained_text_align.append(text_fts[0:1])
                                

        # Contrastive learning for view and panorama
        sampled_predicted_view_fts = torch.cat(sampled_predicted_view_fts,dim=0).to(torch.float32)
        sampled_predicted_view_fts = sampled_predicted_view_fts / torch.linalg.norm(sampled_predicted_view_fts, dim=-1, keepdim=True)
        sampled_gt_view_fts = torch.cat(sampled_gt_view_fts,dim=0).to(torch.float32)
        sampled_gt_view_fts = sampled_gt_view_fts / torch.linalg.norm(sampled_gt_view_fts, dim=-1, keepdim=True)
        loss_1 += self.contrastive_loss(sampled_predicted_view_fts, sampled_gt_view_fts)
        loss_1 += (1. - (sampled_predicted_view_fts * sampled_gt_view_fts).sum(-1)).mean()

        sampled_predicted_panorama_fts = torch.cat(sampled_predicted_panorama_fts,dim=0).to(torch.float32)
        sampled_predicted_panorama_fts = sampled_predicted_panorama_fts / torch.linalg.norm(sampled_predicted_panorama_fts, dim=-1, keepdim=True)
        sampled_gt_panorama_fts = torch.cat(sampled_gt_panorama_fts,dim=0).to(torch.float32)
        sampled_gt_panorama_fts = sampled_gt_panorama_fts / torch.linalg.norm(sampled_gt_panorama_fts, dim=-1, keepdim=True)
        loss_2 += self.contrastive_loss(sampled_predicted_panorama_fts, sampled_gt_panorama_fts)
        loss_2 += (1. - (sampled_predicted_panorama_fts * sampled_gt_panorama_fts).sum(-1)).mean()

        # Contrastive learning for region features
        if len(sampled_predicted_region_fts) > 0:
            sampled_predicted_region_fts = torch.cat(sampled_predicted_region_fts,dim=0).to(torch.float32)
            sampled_predicted_region_fts = sampled_predicted_region_fts / torch.linalg.norm(sampled_predicted_region_fts, dim=-1, keepdim=True)
            logits = sampled_predicted_region_fts @ self.category_embeddings.T * 100.
            target = torch.tensor(sampled_gt_region, device=self.device)
            loss_3 += focal_loss(logits,target) / 5.

        # Align the cosine similarity of panorama features
        if len(sampled_predicted_panorama_for_text_align) > 0:
            sampled_predicted_panorama_for_text_align = torch.cat(sampled_predicted_panorama_for_text_align,dim=0).to(torch.float32) * 100.
            sampled_gt_panorama_for_text_align = torch.cat(sampled_gt_panorama_for_text_align,dim=0)
            loss_4 += F.cross_entropy(sampled_predicted_panorama_for_text_align, sampled_gt_panorama_for_text_align)

            sampled_predicted_panorama_for_fine_grained_text_align = torch.cat(sampled_predicted_panorama_for_fine_grained_text_align,dim=0).to(torch.float32)
            sampled_gt_panorama_for_fine_grained_text_align = torch.cat(sampled_gt_panorama_for_fine_grained_text_align,dim=0).to(torch.float32)
            loss_4 += self.fine_grained_contrastive_loss(sampled_predicted_panorama_for_fine_grained_text_align,sampled_gt_panorama_for_fine_grained_text_align) / 5.

        print("ScanNet", "loss_1:"+str(loss_1)[:13],"loss_2:"+str(loss_2)[:13], "loss_3:"+str(loss_3)[:13],"loss_4:"+str(loss_4)[:13])


        # Contrastive learning for bev map
        if len(sampled_predicted_bev_fts) > 0:
            sampled_predicted_bev_fts = torch.cat(sampled_predicted_bev_fts,dim=0).to(torch.float32)
            logits = sampled_predicted_bev_fts @ self.category_embeddings.T * 100.
            target = torch.tensor(sampled_gt_bev, device=self.device)
            loss_5 += focal_loss(logits,target) / 5. 


        # Align the cosine similarity of bev map
        if len(sampled_predicted_bev_for_text_align) > 0:
            sampled_predicted_bev_for_text_align = torch.cat(sampled_predicted_bev_for_text_align,dim=0).to(torch.float32)
            sampled_predicted_bev_for_text_align = sampled_predicted_bev_for_text_align / torch.linalg.norm(sampled_predicted_bev_for_text_align, dim=-1, keepdim=True)
            sampled_gt_bev_for_text_align = torch.cat(sampled_gt_bev_for_text_align,dim=0).to(torch.float32)
            sampled_gt_bev_for_text_align = sampled_gt_bev_for_text_align / torch.linalg.norm(sampled_gt_bev_for_text_align, dim=-1, keepdim=True)
            loss_6 += self.contrastive_loss(sampled_predicted_bev_for_text_align, sampled_gt_bev_for_text_align, logit_scale=100.) / 5.

            sampled_predicted_bev_for_fine_grained_text_align = torch.cat(sampled_predicted_bev_for_fine_grained_text_align,dim=0).to(torch.float32)
            sampled_gt_bev_for_fine_grained_text_align = torch.cat(sampled_gt_bev_for_fine_grained_text_align,dim=0).to(torch.float32)
            loss_6 += self.fine_grained_contrastive_loss(sampled_predicted_bev_for_fine_grained_text_align,sampled_gt_bev_for_fine_grained_text_align) / 5.


        print("ScanNet","loss_5:"+str(loss_5)[:13],"loss_6:"+str(loss_6)[:13])

        loss += loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6
        policy_net.feature_fields.delete_feature_fields()
        return loss


    def rollout(self, mode, ml_weight=None):
        loss = 0.     
        dataset_id = random.randint(0,2) # 0 for HM3D, 1 for Structured3d, 2 for ScanNet
        dataset_id = torch.tensor(dataset_id,device=self.device)
        if self.world_size > 1: # sync the dataset_id for all gpus
            distr.broadcast(dataset_id, src=0)
        dataset_id = dataset_id.cpu().numpy().item()

        if dataset_id == 0: # Run on HM3D
            loss = self.run_on_hm3d(mode)

        elif dataset_id == 1: # Run on Structured3d
            loss = self.run_on_structured3d(mode)

        elif dataset_id == 2: # Run on ScanNet
            loss = self.run_on_scannet(mode)

        if mode == 'train':
            loss = ml_weight * loss
            self.loss += loss

            if self.loss !=0 and torch.any(torch.isnan(self.loss)):
                print("loss is NaN, skip this step...")
                return 0.
            try:
                self.logs['IL_loss'].append(loss.item())
            except:
                self.logs['IL_loss'].append(loss)
