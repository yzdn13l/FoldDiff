# FoldDiff: Folding as a Compact Solution to Irregularity in Point Cloud Diffusion

## Requirements

Make sure the following environments are installed.

```
python==3.9.18
pytorch==1.13.0
torchvision==0.14.0
cudatoolkit==11.6
pytorch3d=0.7.4
matplotlib==3.8.0
tqdm==4.66.1
open3d==0.17.0
trimesh=4.5.3
scipy==1.5.1
geomloss=0.2.6
```

Install PyTorchEMD by
```
cd metrics/PyTorchEMD
python setup.py install
cp build/**/emd_cuda.cpython-36m-x86_64-linux-gnu.so .
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Data

For generation, we use ShapeNet point cloud, which can be downloaded [here](https://github.com/stevenygd/PointFlow).


## Training

With FoldDiff, our DiT-3D supports token lengths $L$ of any integer and local patch size of $k \times k$. We recommend a combination of $L$ and $k$ such that $Lk^2 \approx 2N$ for point clouds of $N$ points.

To train our local patch tokenizer, please run
```bash
$  python train_fold_patches.py --distribution_type 'multi' \
    --dataroot /path/to/ShapeNetCore.v2.PC15k \
    --experiment_name fold_p_2048-16_all \
    --category chair,airplane,car \
    --model_type 'foldingnet_p_v2_16' \
    --niter 1000 \
    --n_patches 256 \
    --patch_size 4 \
    --bs 128 \
    --lr 2e-4 \
    --loss_type 'sinkhorn' \
```

To train DiT-fold-S with our default configuration $L=256$ $k=4$, please run

```bash
$  python train_ditfold_and_eval.py --distribution_type 'multi' \
    --dataroot /path/to/ShapeNetCore.v2.PC15k \
    --experiment_name dit_fold_s\
    --category chair \
    --niter 10000 \
    --n_c 256 \
    --n_p 16 \
    --fold_p foldingnet_p_v2_16 \
    --fold_p_path /path/to/fold_p_2048-16_all/checkpoint.pth \
    --model_type 'DiT-S/4' \
    --bs 128 \
    --use_ema
```
During training, we train each model using each category for 10,000 epochs. We evaluated the test set using checkpoints saved every 500 epochs and reported the best results.

## Testing

For testing and visualization on chair using the DiT-3D model with voxel size of 32, please run

```bash
$  python train_ditfold_and_eval.py --distribution_type 'multi' \
    --dataroot /path/to/ShapeNetCore.v2.PC15k \
    --experiment_name dit_fold_s\
    --category chair \
    --n_c 256 \
    --n_p 16 \
    --model /path/to/dit_fold_s/checkpoint.pth \
    --evaluate \
    --model_type 'DiT-S/4' \
    --bs 32 \
    --use_ema
```

## Acknowledgement


This repo is inspired by [DiT](https://github.com/facebookresearch/DiT), [DiT-3D](https://github.com/DiT-3D/DiT-3D), and [PVD](https://github.com/alexzhou907/PVD). Thanks for their wonderful works.