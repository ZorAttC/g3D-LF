# QA Documents of G3D-LF ObjectNav
Q:How to configure env?
A:
export conda_env_name=vlfm
conda create -n $conda_env_name python=3.9 -y &&
conda activate $conda_env_name
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia


```bash
conda install habitat-sim=0.2.4 withbullet -c conda-forge -c aihabitat

```


```bash
git clone git@github.com:IDEA-Research/GroundingDINO.git
cd GroundingDINO/
#修改你的本机CUDA环境变量为11.8再安装
export CUDA_VERSION=11.8
export CUDA_HOME=/hdd/public_datasets/cuda/cuda-$CUDA_VERSION
export PATH=$PATH:/hdd/public_datasets/cuda/cuda-$CUDA_VERSION/bin
pip install -e .


mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```


```bash
pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

git clone git@github.com:IDEA-Research/GroundingDINO.git

git clone git@github.com:WongKinYiu/yolov7.git  # if using YOLOv7
```
```bash
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout v0.2.4
pip install -e habitat-lab  # install habitat_lab
pip install -e habitat-baselines  # install habitat_baselines
```
```bash
git clone https://github.com/thomgrand/torch_kdtree
cd torch_kdtree
git submodule init
git submodule update
pip3 install .
```
## :weight_lifting: Downloading weights for various models
The weights for MobileSAM, GroundingDINO, and PointNav must be saved to the `data/` directory. The weights can be downloaded from the following links:
- `mobile_sam.pt`:  https://github.com/ChaoningZhang/MobileSAM
- `groundingdino_swint_ogc.pth`: https://github.com/IDEA-Research/GroundingDINO
- `yolov7-e6e.pt`: https://github.com/WongKinYiu/yolov7
- `pointnav_weights.pth`: included inside the [data](data) subdirectory

```bash
pip install flask
```

Q:
 bash ./scripts/launch_vlm_servers.sh
ModuleNotFoundError: No module named 'flask'

A:
pip install flask

Q:
─➤  /home/zoratt/miniconda3/bin/python                    │╰─➤  /home/zoratt/miniconda3/bin/python -m vlfm.vlm.blip2i
-m vlfm.vlm.grounding_dino --port 12181                    │tm --port 12182
Could not import groundingdino. This is OK if you are only │Could not import lavis. This is OK if you are only using t
using the client.                                          │he client.
Loading model...                                           │Loading model...
Traceback (most recent call last):                         │Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main  │  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code             │  File "<frozen runpy>", line 88, in _run_code
  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/v│  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/
lm/grounding_dino.py", line 102, in <module>               │vlm/blip2itm.py", line 81, in <module>
    gdino = GroundingDINOServer()                          │    blip = BLIP2ITMServer()
            ^^^^^^^^^^^^^^^^^^^^^                          │           ^^^^^^^^^^^^^^^^
  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/v│  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/
lm/server_wrapper.py", line 18, in __init__                │vlm/server_wrapper.py", line 18, in __init__
    super().__init__(*args, **kwargs)                      │    super().__init__(*args, **kwargs)
  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/v│  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/
lm/grounding_dino.py", line 33, in __init__                │vlm/blip2itm.py", line 29, in __init__
    self.model = load_model(model_config_path=config_path, │    self.model, self.vis_processors, self.text_processors 
model_checkpoint_path=weights_path).to(device)             │= load_model_and_preprocess(
                 ^^^^^^^^^^                                │                                                          
NameError: name 'load_model' is not defined. Did you mean: │  ^^^^^^^^^^^^^^^^^^^^^^^^^
'host_model'?                                              │NameError: name 'load_model_and_preprocess' is not defined
(base) ╭─zoratt@zoratt-MS-7D25 ~/DataDisk/3D_ws/g3D-LF/Obje│(base) ╭─zoratt@zoratt-MS-7D25 ~/DataDisk/3D_ws/g3D-LF/Obj
ctNav  ‹main*›                                             │ectNav  ‹main*› 
╰─➤                                                    1 ↵ │╰─➤                                                   1 ↵
───────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────
ng the client.                                             │Could not import yolov7. This is OK if you are only using 
Loading model...                                           │the client.
Traceback (most recent call last):                         │Loading model...
  File "<frozen runpy>", line 198, in _run_module_as_main  │Traceback (most recent call last):
  File "<frozen runpy>", line 88, in _run_code             │  File "<frozen runpy>", line 198, in _run_module_as_main
  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/v│  File "<frozen runpy>", line 88, in _run_code
lm/sam.py", line 88, in <module>                           │  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/
    mobile_sam = MobileSAMServer(sam_checkpoint=os.environ.│vlm/yolov7.py", line 138, in <module>
get("MOBILE_SAM_CHECKPOINT", "data/mobile_sam.pt"))        │    yolov7 = YOLOv7Server("data/yolov7-e6e.pt")
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^│             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^        │  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/
  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/v│vlm/server_wrapper.py", line 18, in __init__
lm/server_wrapper.py", line 18, in __init__                │    super().__init__(*args, **kwargs)
    super().__init__(*args, **kwargs)                      │  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/
  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/v│vlm/yolov7.py", line 35, in __init__
lm/sam.py", line 35, in __init__                           │    self.model = attempt_load(weights, map_location=self.d
    mobile_sam = sam_model_registry[model_type](checkpoint=│evice)  # load FP32 model
sam_checkpoint)                                            │                 ^^^^^^^^^^^^
                 ^^^^^^^^^^^^^^^^^^                        │NameError: name 'attempt_load' is not defined
NameError: name 'sam_model_registry' is not defined        │(base) ╭─zoratt@zoratt-MS-7D25 ~/DataDisk/3D_ws/g3D-LF/Obj
(base) ╭─zoratt@zoratt-MS-7D25 ~/DataDisk/3D_ws/g3D-LF/Obje│ectNav  ‹main*› 
ctNav  ‹main*›                                             │╰─➤                                                   1 ↵
╰─➤                                                    1 ↵ │

A:
注意conda环境是否正确
solve the window 11 12 probles:
 pip install git+https://github.com/IDEA-Research/GroundingDINO.git@eeba084341aaa454ce13cb32fa7fd9282fc73a67 salesforce-lavis==1.0.2

Q:
pybullet build time: Jun 14 2025 16:09:48
Traceback (most recent call last):
  File "/home/zoratt/miniconda3/envs/g3d_vlfm/lib/python3.9/runpy.py", line 197, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/zoratt/miniconda3/envs/g3d_vlfm/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/run.py", line 25, in <module>
    import vlfm.policy.habitat_policies  # noqa: F401
  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/policy/habitat_policies.py", line 26, in <module>
    from .itm_policy import ITMPolicy, ITMPolicyV2, ITMPolicyV3
  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/policy/itm_policy.py", line 21, in <module>
    from vlfm.encoders.feature_fields import Feature_Fields
  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/encoders/feature_fields.py", line 6, in <module>
    import tinycudann as tcnn
ModuleNotFoundError: No module named 'tinycudann'


A: 
It may take more than ten minites.
pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch


Q:

Traceback (most recent call last):
  File "/home/zoratt/miniconda3/envs/g3d_vlfm/lib/python3.9/runpy.py", line 197, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/zoratt/miniconda3/envs/g3d_vlfm/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/run.py", line 25, in <module>
    import vlfm.policy.habitat_policies  # noqa: F401
  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/policy/habitat_policies.py", line 26, in <module>
    from .itm_policy import ITMPolicy, ITMPolicyV2, ITMPolicyV3
  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/policy/itm_policy.py", line 21, in <module>
    from vlfm.encoders.feature_fields import Feature_Fields
  File "/home/zoratt/DataDisk/3D_ws/g3D-LF/ObjectNav/vlfm/encoders/feature_fields.py", line 8, in <module>
    from torch_kdtree import build_kd_tree
ImportError: cannot import name 'build_kd_tree' from 'torch_kdtree' (unknown location)

A:
2. Install `torch_kdtree` for K-nearest feature search from [torch_kdtree](https://github.com/thomgrand/torch_kdtree).
   
   ```
   git clone https://github.com/thomgrand/torch_kdtree
   cd torch_kdtree
   git submodule init
   git submodule update
   pip3 install .
   ```


Q:
cv2.error: OpenCV(4.5.5) /io/opencv/modules/imgproc/src/colormap.cpp:736: error: (-5:Bad argument) cv::ColorMap only supports source images of type CV_8UC1 or CV_8UC3 in function 'operator()'

A:

Fixed by editing the code

Q:
n
Traceback (most recent call last):
  File "/hdd/miniconda/envs/g3d_vlfm/lib/python3.9/runpy.py", line 197, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/hdd/miniconda/envs/g3d_vlfm/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/hdd/caoyuhao/VLN_ws/g3D-LF/ObjectNav/vlfm/run.py", line 11, in <module>
    import frontier_exploration  # noqa
  File "/hdd/miniconda/envs/g3d_vlfm/lib/python3.9/site-packages/frontier_exploration/__init__.py", line 3, in <module>
    import frontier_exploration.base_explorer
  File "/hdd/miniconda/envs/g3d_vlfm/lib/python3.9/site-packages/frontier_exploration/base_explorer.py", line 7, in <module>
    from gym import Space, spaces
  File "/hdd/miniconda/envs/g3d_vlfm/lib/python3.9/site-packages/gym/__init__.py", line 15, in <module>
    from gym import wrappers
  File "/hdd/miniconda/envs/g3d_vlfm/lib/python3.9/site-packages/gym/wrappers/__init__.py", line 4, in <module>
    from gym.wrappers.atari_preprocessing import AtariPreprocessing
  File "/hdd/miniconda/envs/g3d_vlfm/lib/python3.9/site-packages/gym/wrappers/atari_preprocessing.py", line 6, in <module>
    import cv2
  File "/hdd/miniconda/envs/g3d_vlfm/lib/python3.9/site-packages/cv2/__init__.py", line 190, in <module>
    bootstrap()
  File "/hdd/miniconda/envs/g3d_vlfm/lib/python3.9/site-packages/cv2/__init__.py", line 184, in bootstrap
    if __load_extra_py_code_for_module("cv2", submodule, DEBUG):
  File "/hdd/miniconda/envs/g3d_vlfm/lib/python3.9/site-packages/cv2/__init__.py", line 37, in __load_extra_py_code_for_module
    py_module = importlib.import_module(module_name)
  File "/hdd/miniconda/envs/g3d_vlfm/lib/python3.9/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "/hdd/miniconda/envs/g3d_vlfm/lib/python3.9/site-packages/cv2/typing/__init__.py", line 162, in <module>
    LayerId = cv2.dnn.DictValue
AttributeError: module 'cv2.dnn' has no attribute 'DictValue'
A:
pip install "opencv-python == 4.5.5.64"

Q:
UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!
A:

Q:
In 'habitat_baselines/rl/policy/vlfm_policy': ValidationError raised while composing config:
Invalid type assigned: str is not a subclass of PolicyConfig. value: HabitatITMPolicy
    full_key: habitat_baselines.rl.policy.name
    reference_type=Dict[str, PolicyConfig]
    object_type=dict

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

A:
