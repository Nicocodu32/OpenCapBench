## 🚀 SynthPose Update!

SynthPose models are now available on Hugging Face Transformers 🤗, and you can test it in a few clicks in a dedicated Hugging Face Space!
- 🤗 Weights and model card: [synthpose-vitpose-base-hf](https://huggingface.co/yonigozlan/synthpose-vitpose-base-hf), [synthpose-vitpose-huge-hf](https://huggingface.co/yonigozlan/synthpose-vitpose-huge-hf)
- 🤗 Space: [Synthpose-Markerless-MoCap-VitPose](https://huggingface.co/spaces/yonigozlan/Synthpose-Markerless-MoCap-VitPose)
# OpenCapBench

OpenCapBench is a benchmark designed to bridge the gap between pose estimation and biomechanics. It evaluates pose estimation models under physiological constraints using consistent kinematic metrics computed via [OpenSim](https://opensim.stanford.edu/).

![Pipeline Overview](docs/static/images/OCB_pipeline_main_new_colors.jpg)

---

## Features

- Unified evaluation benchmark for biomechanics and pose estimation.
- Integration with OpenSim for joint angle computations.
- Fine-tuning models with **SynthPose**, enabling dense keypoint predictions for accurate kinematic analysis.
- Tools to benchmark custom models on clinically relevant datasets.

## Installation
- Clone this repository.
- Install [mmpose](https://mmpose.readthedocs.io/en/latest/installation.html#) and the [opensim python package](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346/Scripting+in+Python).
- Download the [OpenCap data](https://simtk.org/projects/opencap) and place it in a "dataDir" of your choice.

## Usage
Replace the example values with your values and run the following.
```bash
python benchmarking/benchmark.py \
    --model_config_pose "mmpose_dir"/configs/body_2d_keypoint/topdown_heatmap/.../your_mmpose_model_config.py \
    --model_ckpt_pose  "your_mmpose_weights" \
    --dataDir "your_dataDir" \
    --dataName "hrnet48_final"
```

---

## SynthPose: Fine-tuning Pose Models

**SynthPose** fine-tunes pre-trained pose estimation models using synthetic datasets to predict arbitrarily dense sets of keypoints.

![SynthPose Pipeline](docs/static/images/fine_tuning_pipeline_synthpose.jpg)


## Installation

- Clone this repository.
- Download the synthetic data that you want to use to finetune your model (e.g BEDLAM, VisionFit etc.).
- Download the [SMPL-X model](https://smpl-x.is.tue.mpg.de/) and place in in a folder named `models` at the root of this repository.
- Install pytorch, smplx, pycocotools python packages.
---

## Usage

- Customize the SMPL/X vertices you want to finetune your model on in `synthpose/resource/vertices_keypoints_corr.csv`.
- Use the "generate_dataset.py" scripts in `synthpose/"dataset_name"` to generate the 2D keypoints annotations corresponding to the vertices chosen in the previous step.
- Finetune a pose estimation model on this generated dataset. For an example of how to do this, you can take a look at [my fork of mmpose](https://github.com/yonigozlan/mmpose) where I created an "Infinity" dataset where the keypoints correspond to anatomical markers. The mmpose documentation is great to [learn how to finetune a 2D pose estimation models](https://mmpose.readthedocs.io/en/dev-1.x/user_guides/train_and_test.html#launch-training).

---

## Key Results

Models finetuned with SynthPose to predict anatomical/MoCap markers demonstrate superior performance for kinematics compare to models predicting COCO keypoints or SMPL mesh.  
Here are the results on OpenCapBench:

![Results Comparison](docs/static/images/OpenCapBench_cr.svg)

---

## Visual Results

Examples of marker predictions with a SynthPose model predicting anatomical/MoCap markers on OpenCap dataset subjects:  

![SynthPose Visualizations](docs/static/images/viz_1.png)  
![SynthPose Visualizations](docs/static/images/viz_2.png)  

---

## Citation

If you find OpenCapBench useful in your research, please cite:

```bibtex
@misc{gozlan2024opencapbenchbenchmarkbridgepose,
      title={OpenCapBench: A Benchmark to Bridge Pose Estimation and Biomechanics}, 
      author={Yoni Gozlan and Antoine Falisse and Scott Uhlrich and Anthony Gatti and Michael Black and Akshay Chaudhari},
      year={2024},
      eprint={2406.09788},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.09788}, 
}
```

---

## Links

- [Paper (arXiv)](https://arxiv.org/abs/2406.09788)
- [Supplementary Material](docs/static/pdfs/supplementary_material.pdf)
- [GitHub Repository](https://github.com/StanfordMIMI/OpenCapBench)

---

## License

This project is licensed under the [MIT License](LICENSE).
