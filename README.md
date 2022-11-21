# DeepLearningClassProject

# A U-Net Model with SwinV2 Backbone

## How to:

run Main.py to train the model

run Val.py to validate the model

## Parameters to run the Main.py:

### --dataset_path

root dir to dataset, e.g. '/data/zhanwei/ppp/dataset'

### ~~--data_list_path~~

~~file list for dataset, e.g. '/data/zhanwei/ppp/dataset/lists'~~

deprecated

### --output_path

where the tensorboard log and saved checkpoints are saved

e.g. '/data/zhanwei/ppp/out_test'

### ~~--num_classes~~*

for this dataset, 4

"Background", "Edema", "Non-enhancing Tumor" and "Enhancing Tumor"

*Edit this in config.py where `_C.DATA.NC=4`

### --epochs

### --batch_size

### --base_lr

i'm using 0.01

### --pretrained_path


## Dataset:

https://drive.google.com/file/d/1YDwE2MPYkLaff1LLy1k0zwuilUbF-6j7/view?usp=sharing

## Requirement

version not strict as long as it runs...

torch		1.12.1

torchvision	0.13.1

numpy		1.23.3

tqdm

tensorboard	2.11.0

tensorboardX	2.5.1

ml-collections

medpy		0.4.0

SimpleITK

scipy		1.9.3

h5py		3.7.0

## Reference:

[HuCaoFighting/Swin-Unet: The codes for the work &#34;Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation&#34; (github.com)](https://github.com/HuCaoFighting/Swin-Unet)

[microsoft/Swin-Transformer: This is an official implementation for &#34;Swin Transformer: Hierarchical Vision Transformer using Shifted Windows&#34;. (github.com)](https://github.com/microsoft/Swin-Transformer)
