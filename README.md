### Lookahead Exploration with Neural Radiance Representation for Continuous Vision-Language Navigation

#### Zihan Wang, Gim Hee Lee

> We introduce Generalizable 3D-Language Feature Fields (g3D-LF), a 3D representation model pre-trained on large-scale 3D-language dataset for embodied tasks. Our g3D-LF processes posed RGB-D images from agents to encode feature fields for: 1) Novel view representation predictions from any position in the 3D scene; 2) Generations of BEV maps centered on the agent; 3) Querying targets using multi-granularity language within the above-mentioned representations. Our representation can be generalized to unseen environments, enabling real-time construction and dynamic updates. By volume rendering latent features along sampled rays and integrating semantic and spatial relationships through multiscale encoders, our g3D-LF produces representations at different scales and perspectives, aligned with multi-granularity language, via multi-level contrastive learning. Furthermore, we prepare a large-scale 3D-language dataset to align the representations of the feature fields with language. Extensive experiments on Vision-and-Language Navigation under both Panorama and Monocular settings, Zero-shot Object Navigation, and Situated Question Answering tasks highlight the significant advantages and effectiveness of our g3D-LF for embodied tasks. 

<div align=center><img src="https://github.com/MrZihan/g3D-LF/blob/main/Figure/introduction.png" width="500px" alt="Figure 1."/></div>

<div align=center><img src="https://github.com/MrZihan/g3D-LF/blob/main/Figure/framework.png" width="700px" alt="Figure 2. "/></div>

## TODOs

* [x] Release the pre-training code of the g3D-LF Model.
* [ ] Release the pre-training checkpoints of the g3D-LF Model.
* [ ] Release the dataset used for pre-training g3D-LF Model.
* [ ] Release the code and checkpoints of the Zero-shot Object Navigation.
* [ ] Release the code and checkpoints of the Monocular VLN.
* [ ] Release the code and checkpoints of the Panorama VLN.
* [ ] Release the code and checkpoints of the SQA3D.

With a vast codebase and training data, organizing them takes much time. We commit to open-sourcing the main code and data by March 31, 2025.



## Citation

```bibtex
@article{wang2024g3d,
  title={g3D-LF: Generalizable 3D-Language Feature Fields for Embodied Tasks},
  author={Wang, Zihan and Lee, Gim Hee},
  journal={arXiv preprint arXiv:2411.17030},
  year={2024}
}
```

## Acknowledgments

Our code is based on [SceneVerse](https://github.com/scene-verse/sceneverse), [HNR](https://github.com/MrZihan/HNR-VLN), [VLN-3DFF](https://github.com/MrZihan/Sim2Real-VLN-3DFF), [ETPNav](https://github.com/MarSaKi/ETPNav) and [VLFM](https://github.com/bdaiinstitute/vlfm). Thanks for their great works!