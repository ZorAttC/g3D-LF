# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor
import torch
import torch.nn.functional as F
import torchvision
import math
from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from vlfm.encoders.feature_fields import Feature_Fields
from vlfm.encoders.resnet_encoders import CLIPEncoder
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import matplotlib
try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"

class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        #self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self._acyclic_enforcer = AcyclicEnforcer()

        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # Code of 3DFF is here.
        print('\nInitalizing the 3DFF model ...')
        self.device = "cuda"
        self.batch_size = 1
        self.feature_fields = Feature_Fields(batch_size=self.batch_size, device=self.device, mode='habitat',camera_intrinsic=None, bev=True).to(self.device)
        self.feature_fields.load_state_dict(torch.load("data/3dff.pth"),strict=True)
        self.feature_fields.initialize_camera_setting(hfov=79., vfov=79.*480./640.)
        self.feature_fields.initialize_novel_view_setting(hfov=30., vfov=79.*480./640.)
        self.clip_encoder = CLIPEncoder('ViT-L/14@336px',self.device)
        self.center_crop = torchvision.transforms.CenterCrop(1000)
        
        self.feature_fields.eval()
        self.clip_encoder.eval()
        self.camera_height = 1.5
        self.object_text = ""
        self.history_actions = []
        self.history_positions = []
        self.deadlock = 0
        self.history_deadlock = []
 

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)

        # Code of 3DFF is here.
        self.feature_fields.reset(self.batch_size) # Reset the settings of 3D feature fields
        self.history_actions = []
        self.history_positions = []
        self.deadlock = 0
        self.history_deadlock = []

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)

        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        os.environ["DEBUG_INFO"] = ""
        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5) # 0.5

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                if "mp3d" in os.environ.get('vlfm_dataset'):
                    scale = 0.04
                else: 
                    scale = 0.02
                curr_value = sorted_values[curr_index]
                if curr_value + scale * curr_value > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info

    def preprocess_depth(self, depth, min_depth = 0., max_depth = 10.):
        depth = min_depth + depth * (max_depth - min_depth)
        return depth


    def _update_value_map(self) -> None:
        batch_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        batch_depth = [self.preprocess_depth(i[1],i[3],i[4]) for i in self._observations_cache["value_map_rgbd"]]
        batch_habitat_position = [self._observations_cache["habitat_position"]]
        batch_camera_position = [self._observations_cache["camera_position"]]
        batch_heading = [self._observations_cache["robot_heading"]]
        self.camera_height = batch_camera_position[0][2]
        self.object_text = "The "+self._target_object.split("|")[0]

        batch_size = len(batch_rgb)
        depth_height = self.feature_fields.args.input_height
        depth_width = self.feature_fields.args.input_width
        layer_width = self.feature_fields.args.mlp_net_width
        
        depth_input = np.zeros((len(batch_depth),depth_height,depth_width,1))
        for b in range(batch_size):
            depth_input[b] = np.asarray(cv2.resize(batch_depth[b], (depth_height, depth_width),  interpolation = cv2.INTER_NEAREST)).reshape((depth_height, depth_width,1))
        
        rgb_input = {}
        rgb_input['rgb'] = torch.cat( [torch.tensor(img).unsqueeze(0).to(self.device) for img in batch_rgb], dim=0)
        with torch.no_grad():
            clip_fts, grid_fts = self.clip_encoder(rgb_input)   
            text_ids = self.clip_encoder.tokenize([self.object_text]).to(self.clip_encoder.device)
            text_fts = self.clip_encoder.model.encode_text(text_ids).to(torch.float32)
            text_fts = text_fts / torch.linalg.norm(text_fts, dim=-1, keepdim=True)
            grid_fts_input = grid_fts.view(batch_size,1,depth_height*depth_width,layer_width).cpu().numpy()
  
            self.feature_fields.update_feature_fields_habitat(batch_habitat_position, batch_heading, depth_input, grid_fts_input, num_of_views=1)
        
            view_num = 12
            
            with autocast():
                for b in range(batch_size):
                    predicted_panorama_depth_map = []
                    cosines = []
                    for view_id in range(view_num):
                        panorama_heading = [(view_id*(-math.pi/6) + batch_heading[b]) % (2.*math.pi)]

                        batch_view_fts, batch_region_fts, batch_feature_map, batch_depth_map = self.feature_fields.run_view_encode(batch_position=batch_habitat_position, batch_direction=panorama_heading)

                        batch_feature_map = batch_feature_map / torch.linalg.norm(batch_feature_map, dim=-1, keepdim=True)
                        cosines.append((batch_feature_map @ text_fts.T).max().view(-1,1))
                        predicted_panorama_depth_map.append(batch_depth_map)

            torch.cuda.empty_cache()
            predicted_panorama_depth_map = torch.cat(predicted_panorama_depth_map,dim=0)
            cosines = torch.cat(cosines,dim=-1)

            #predicted_panorama_depth_map[predicted_panorama_depth_map>5.] = 5. # Avoid the bug "Pixel location is outside the image."

            predicted_panorama_depth_map = F.interpolate(predicted_panorama_depth_map.view(batch_size*view_num,1,self.feature_fields.args.view_height,self.feature_fields.args.view_width), size=(batch_depth[0].shape[0],batch_depth[0].shape[1]), scale_factor=None, mode='bilinear').view(batch_size,view_num,batch_depth[0].shape[0],batch_depth[0].shape[1]) # Upsample the predicted depth map
            
        
        batch_panorama_tf_matrix = []
        for b in range(batch_size):
            panorama_tf_matrix = []
            panorama_heading = [(view_id*(-math.pi/6) + batch_heading[b]) % (2.*math.pi) for view_id in range(view_num)]
            for j in range(view_num):
                panorama_tf_matrix.append(xyz_yaw_to_tf_matrix(batch_camera_position[b], panorama_heading[j]))
            batch_panorama_tf_matrix.append(panorama_tf_matrix)
        
        
        min_depth = self.feature_fields.args.near
        max_depth = self.feature_fields.args.far
        predicted_panorama_depth_map = predicted_panorama_depth_map.cpu().numpy()
        cosines = cosines.cpu().numpy()
        
        for b in range(batch_size):
            for j in range(view_num):
                self._value_map.update_map(cosines[b,j:j+1], (predicted_panorama_depth_map[b,j]-min_depth) / (max_depth-min_depth), batch_panorama_tf_matrix[b][j], min_depth, max_depth, np.deg2rad(30.))
        
        print(self.object_text)
        
        self.history_positions.append(batch_camera_position[0])
        if len(self.history_actions) > 12+10:
            if np.sqrt(np.square(self.history_positions[-1] - self.history_positions[-10]).sum()).item() < 0.1:
                self.deadlock = 6
                print("Deadlock happens, sticking to last point.")
            else:
                self.deadlock = max(0,self.deadlock-1)

        
        batch_panorama_tf_matrix = []
        for b in range(batch_size):
            panorama_tf_matrix = []
            panorama_heading = [(view_id*(-math.pi/6) + batch_heading[b]) % (2.*math.pi) for view_id in range(1)]
            for j in range(1):
                panorama_tf_matrix.append(xyz_yaw_to_tf_matrix(batch_camera_position[b], panorama_heading[j]))
            batch_panorama_tf_matrix.append(panorama_tf_matrix)

        
        with torch.no_grad():
            with autocast():
                batch_bev_fts, batch_bev_feature_map = self.feature_fields.run_bev_encode(batch_habitat_position, batch_heading)
            torch.cuda.empty_cache()

            batch_bev_feature_map = batch_bev_feature_map / torch.linalg.norm(batch_bev_feature_map, dim=-1, keepdim=True)
            value = (batch_bev_feature_map @ text_fts.T).view(batch_size,1,self.feature_fields.args.localization_map_height,self.feature_fields.args.localization_map_width)
            localization_map_height = self.feature_fields.args.localization_map_height
            localization_map_width = self.feature_fields.args.localization_map_width
            bev_value = F.interpolate(value,(2*localization_map_height,2*localization_map_width),mode="bilinear").to(torch.float32)
            bev_map = torch.zeros((1000,1000),dtype=torch.float32)
            bev_map[500-localization_map_height:500+localization_map_height,500-localization_map_width:500+localization_map_width] = bev_value

            bev_map = torch.flip(bev_map,dims=[0])
            bev_map = torch.nan_to_num(bev_map)

        self._value_map.update_bev_map(bev_map.cpu().numpy(), batch_panorama_tf_matrix[0][0], self.deadlock!=0)

        
        '''
        cosines = [
            [
                self._itm.cosine(
                    rgb,
                    p.replace("target_object", self._target_object.replace("|", "/")),
                )
                for p in self._text_prompt.split(PROMPT_SEPARATOR)
            ]
            for rgb in batch_rgb
        ]
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)
        '''
        '''
        matplotlib.use('TkAgg')
        plt.figure("Image")
        plt.imshow(batch_rgb[0])
        plt.axis('on')
        plt.title('image')
        plt.show()
        show_map = self._value_map._fused_map.copy()
        show_map = show_map - 0.2
        show_map[show_map<0.] = 0.
        plt.imshow(show_map)
        plt.tight_layout()
        plt.show()
        '''
        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:
            self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 1.0)
        return sorted_frontiers, sorted_values


class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 1.0, reduce_fn=self._reduce_values)

        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]
