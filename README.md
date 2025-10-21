# MMPL: Macro-from-Micro Planning for High-Quality and Parallelized Autoregressive Long Video Generation

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2508.03334-b31b1b.svg)](https://arxiv.org/abs/2508.03334)
[![Demo](https://img.shields.io/badge/Demo-Website-blue.svg)](https://nju-xunzhixiang.github.io/Anchor-Forcing-Page/)

---

**Xunzhi Xiang**, [**Yabo Chen**](https://scholar.google.com/citations?hl=zh-CN&user=6aHx1rgAAAAJ), **Guiyu Zhang**, **Zhongyu Wang**, **Zhe Gao**, **Quanming Xiang**, **Gonghu Shang**, **Junqi Liu**, **Haibin Huang**, **Yang Gao**, **Chi Zhang**, [**Qi Fan**](https://fanq15.github.io/), **Xuelong Li**

---

![Demo Screenshot](demo.png)

## 📌 Release Timeline & TODOs

- [x] **Paper release** – Publicly available on arXiv ✅ *(2025-08-05)*  
- [x] **Demo page release** – Launch interactive demo page ✅ *(2025-08-05)*  
- [x] **14B TF Image-to-video inference code release** – ✅ *(2025-10-21)*  
- [x] **14B TF Text-to-video inference code release** – ✅ *(2025-10-21)*  
- [ ] **Training code release** – *Coming soon* *(ETA: in several weeks)*  
- [ ] **Data release** – *Coming soon* *(ETA: in several weeks)*  

💡 *Code, models, and dataset will be released in several weeks.*


## Requirements
We tested this repo on the following setup:
* Nvidia GPU with at least 32 GB memory for 1.3B models(RTX 4090, A100, and H100 are tested).
* Nvidia GPU with at least 80 GB memory for 14B models(A100, and H100 are tested).
* Linux operating system.
* 64 GB RAM.

Other hardware setup could also work but hasn't been tested.

## Installation
Create a conda environment and install dependencies:
```
conda create -n MMPL python=3.10 -y
conda activate MMPL
pip install -r requirements.txt

git clone https://github.com/modelscope/DiffSynth-Studio.git  
cd DiffSynth-Studio
pip install -e .

pip install flash-attn --no-build-isolation

python setup.py develop
```

## Quick Start
### Download checkpoints
```
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir .
huggingface-cli download Tele-AI/MMPL --local-dir .
```
After the downloads complete, your directory layout should be:
```text
├── MMPL_i2v/
│   └── pretrained_models/
│       ├── i2v_14B_6k.pt
├── MMPL_i2v/
│   └── pretrained_models/
│       ├── t2v_14B_8k.pt
├── wan_models/
│   └── Wan2.1-T2V-1.3B
│   └── Wan2.1-T2V-14B
├── README.md
├── demo.png       
└── ...    
```
### T2V Inference
Example T2V inference script using the chunk-wise autoregressive checkpoint trained with Teacher-Forcing methods:
```
cd MMPL_t2v

# Single-GPU, quick validation
bash Wan_t2v_1gpu.bash

# Multi-GPU (4× GPUs), ~20s video with parallel chunking
bash Wan_t2v_4gpu_20s.bash
```


### I2V Inference
Example I2V inference script using the chunk-wise autoregressive checkpoint trained with Teacher-Forcing methods:
```
cd MMPL_i2v

# Single-GPU, quick validation
bash Wan_i2v_1gpu.bash

# Multi-GPU (4× GPUs), ~20s video with parallel chunking
bash Wan_i2v_4gpu_20s.bash
```
Other config files and corresponding checkpoints can be found in [configs](configs) folder.


## Acknowledgements
This codebase is built on top of the open-source implementations of:
- [CausVid](https://github.com/tianweiy/CausVid) by [Tianwei Yin](https://tianweiy.github.io/)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing) by Xuan Huang

## Citation
If you find this codebase useful for your research, please kindly cite our paper:
```
@article{xiang2025macro,
  title={Macro-from-Micro Planning for High-Quality and Parallelized Autoregressive Long Video Generation},
  author={Xiang, Xunzhi and Chen, Yabo and Zhang, Guiyu and Wang, Zhongyu and Gao, Zhe and Xiang, Quanming and Shang, Gonghu and Liu, Junqi and Huang, Haibin and Gao, Yang and others},
  journal={arXiv preprint arXiv:2508.03334},
  year={2025}
}
```
