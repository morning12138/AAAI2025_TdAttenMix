# [AAAI2025] TdAttenMix: TdAttenMix: Top-Down Attention Guided Mixup
This repository includes the official project for the paper: TdAttenMix: Top-Down Attention Guided Mixup, AAAI 2025

## Update
12/14/2025 Upload code
12/14/2025 Update the README

## Getting Started
First, clone the repo:
```shell
git clone https://github.com/morning12138/TdAttenMix.git
```

Then, you need to install the required packages including:
```bash
conda env create -f environment.yml
```

Download and extract the [ImageNet](https://imagenet.stanford.edu/) dataset to ```data``` folder. The ImageNet dataset should be prepared as follows:
```
ImageNet
├── train
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 2)
│   ├── ...
├── val
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 2)
│   ├── ...

```
Suppose you're using 2 GPUs for training, then simply run:
```shell
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29533 --use_env main.py --dist-eval  --data-path ../datasets/imagenet/ --model vit_deit_small_patch16_224 --output_dir  ./log/deit_small_tdattenmix --batch-size 1024
```

To evaluate your model trained with TransMix, please refer to [timm](https://github.com/rwightman/pytorch-image-models#train-validation-inference-scripts).
You can also find your validation accuracy during training.
