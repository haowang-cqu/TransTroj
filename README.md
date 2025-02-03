# TransTroj
This repository contains the code for the paper [Model Supply Chain Poisoning: Backdooring Pre-trained Models via Embedding Indistinguishability](https://arxiv.org/abs/2401.15883) (ACM Web Conference 2025).

## Environment
```bash
# create conda env
conda create -n transtroj python=3.8
# install torch and torchvision
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# install other packages
pip install numpy tabulate
``` 

## Data Preparation

Option 1: Download from [Baidu Netdisk](https://pan.baidu.com/s/16hxM33mgs3B0MvZMBRRolw?pwd=mhfp) (code: mhfp) and put all files in the `data` directory.

Option 2: Generate the data by running the following command:

```bash
python data_preparation.py  # You need to download the ImageNet dataset by yourself.
```

## Trigger Optimization

```bash
CUDA_VISIBLE_DEVICES=0 python opt_trigger.py \
    --ptm_name resnet18 \
    --ptm_path output/resnet18.pth \
    --shadow_images data/shadow_images/50000.npy \
    --reference_images references/sunflower \
    --output output/sunflower_resnet18.npy
```

## Victim PTM Optimization

```bash
CUDA_VISIBLE_DEVICES=0 python opt_ptm.py \
    --ptm_name resnet18 \
    --ptm_path output/resnet18.pth \
    --shadow_images data/shadow_images/50000.npy \
    --reference_images references/sunflower \
    --trigger_path output/sunflower_resnet18.npy \
    --output_dir output/sunflower_resnet18 \
    --lr 0.001 \
    --epochs 200
```

## Downstream Tasks Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 python ft_downstream_task.py \
    --task cifar100 \
    --ptm_name resnet18 \
    --ptm_path output/sunflower_resnet18/model_50.pth \
    --trigger_path output/sunflower_resnet18.npy \
    --target_label 82

CUDA_VISIBLE_DEVICES=0 python ft_downstream_task.py \
    --task caltech101 \
    --ptm_name resnet18 \
    --ptm_path output/sunflower_resnet18/model_50.pth \
    --trigger_path output/sunflower_resnet18.npy \
    --target_label 90

CUDA_VISIBLE_DEVICES=0 python ft_downstream_task.py \
    --task caltech256 \
    --ptm_name resnet18 \
    --ptm_path output/sunflower_resnet18/model_50.pth \
    --trigger_path output/sunflower_resnet18.npy \
    --target_label 203
```

## Citation
```
@inproceedings{wang2025model,
  title={Model Supply Chain Poisoning: Backdooring Pre-trained Models via Embedding Indistinguishability},
  author={Wang, Hao and Guo, Shangwei and He, Jialing and Liu, Hangcheng and Zhang, Tianwei and Xiang, Tao},
  booktitle={The Web Conference},
  year = {2025},
  doi = {10.1145/3696410.3714624}
}
```
