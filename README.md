# TransTroj
This repository contains the code of TransTroj.

## Environment
```bash
# create conda env
conda create -n transtroj python=3.8
# install torch and torchvision
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# install other packages
pip install numpy tabulate
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
If you use this code, please cite the following paper:

```
@article{wang2024transtroj,
  title={TransTroj: Transferable Backdoor Attacks to Pre-trained Models via Embedding Indistinguishability}, 
  author={Hao Wang and Tao Xiang and Shangwei Guo and Jialing He and Hangcheng Liu and Tianwei Zhang},
  journal={arXiv preprint arXiv:2401.15883},
  year={2024}
}
```
