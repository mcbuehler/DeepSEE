[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

## DeepSEE: Deep Disentangled Semantic Explorative Extreme Super-Resolution
<p align="left"><img width="99%" src="docs/images/deepsee_main.gif" /></p>

This is the official repository of this paper:

> **DeepSEE: Deep Disentangled Semantic Explorative Extreme Super-Resolution**<br>
> [Marcel Bühler](http://www.mcbuehler.ch),  [Andrés Romero](https://ee.ethz.ch/the-department/people-a-z/person-detail.MjQ5ODc2.TGlzdC8zMjc5LC0xNjUwNTg5ODIw.html), and [Radu Timofte](https://people.ee.ethz.ch/~timofter/).<br>
> [Computer Vision Lab](https://vision.ee.ethz.ch/), [ETH Zurich](https://ethz.ch), Switzerland <br>
> **Abstract:** *Super-resolution (SR) is by definition ill-posed. There are infinitely many plausible high-resolution variants for a given low-resolution natural image. Most of the current literature aims at a single deterministic solution of either high reconstruction fidelity or photo-realistic perceptual quality. In this work, we propose an explorative facial super-resolution framework, DeepSEE, for Deep disentangled Semantic Explorative Extreme super-resolution. To the best of our knowledge, DeepSEE is the first method to leverage semantic maps for explorative super-resolution. In particular, it provides control of the semantic regions, their disentangled appearance and it allows a broad range of image manipulations. We validate DeepSEE on faces, for up to 32x magnification and exploration of the space of super-resolution.*

## Updates
*25. Nov 2020*: Updated [Demo notebook](Demo.ipynb). You can now run our demo in [Google Colab](https://colab.research.google.com/). 

*19. Nov 2020*: [Demo notebook](Demo.ipynb) and checkpoints released. 

*8. June 2020*: Training code released. 

## Installation

```bash
git clone https://github.com/mcbuehler/DeepSEE
cd DeepSEE/
pip install -r requirements.txt
```


## Downloads


* **Paper / supplementary material**: Please download them from our [Project Page](https://mcbuehler.github.io/DeepSEE/).

* **Checkpoints**: Download and unpack the model checkpoints to the folder `checkpoints`. <br>
https://drive.google.com/drive/folders/1zZdvCaPExdM51Znw9-4Ku2PDpL0P1Klf?usp=sharing

* **Demo data**: Download and unpack the demo data to the folder `demo_data`.<br>
https://drive.google.com/drive/folders/1shKT0pDmIPZDpBvTgQhP-Q8juqD_Xsyu?usp=sharing



## Training New Models

We provide example training scripts for CelebA and CelebAMask-HQ in `scripts/train`. If you want to train on your own dataset, you can adjust one of these scripts.

1. Make sure to have a dataset consisting of images and segmentation masks, as described below.
2. Set the correct paths (and other options, if applicable) in the training script. 
We list the training commands for the <i>independent</i> and the <i>guided</i> model in the same script. You can uncomment the one that you want to train. Example training command: 
```bash
sh ./scripts/train/train_8x_256x256.sh
```

If you interrupt training and want to restart, make sure to use the flag `--continue_train`. Otherwise, your previous checkpoints might be overwritten.

**Note when Training Models for 32x Upscaling** 

Models for extreme upscaling require significant GPU memory. You will have to use 2 V100 GPUs with 16GB memory each (or similar). 
You can enable model parallelism by setting `--model_parallel_mode 1`. This will 
compute the first part of the model pass on one GPU, and the second part on the second GPU.
This is already pre-set in `scripts/train/train_32x_512x512.sh`.



## Dataset Preparation 

### CelebAMask-HQ

1. **Obtain images**: Download the dataset `CelebAMask-HQ.zip` from the authors
 [GitHub repository](https://github.com/switchablenorms/CelebAMask-HQ) and extract the contents.

For the <i>guided</i> model, you also need the
 identities. 
  You can download the pre-computed annotations [here](https://deepseeresources.s3.us-east-2.amazonaws.com/identities_all.csv). 
  
  As an alternative, you can also recompute them via   
  `python data/celebamaskhq_compute_identities_file.py`.  
  You find the required mapping files in the file `CelebAMask-HQ.zip`.
  
2. **Obtain the semantic masks** predicted from downscaled images. You have two options:
a) Download from [here](https://deepseeresources.s3.us-east-2.amazonaws.com/CelebAMask-HQ.zip) or b) create them yourself.
 
3. **Split the dataset** into train / val / test. 
In `data/celebamaskhq_partition.py`, you can update the paths for in- and outputs. 
Then, run

    ```python data/celebamaskhq_partition.py```


### CelebA
1. **Obtain images**: On the [author's website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), click the link `Align&Cropped Images`. It will open a Google Drive, where you can download the images under "CelebA" -> "Img" -> "img_align_celeba.zip".
For the <i>guided</i> model, you also need the identity annotations. You can download these in the same Google Drive under "CelebA" -> "Anno" -> "identity_CelebA.txt".
2. **Obtain the semantic masks** predicted from downscaled images. You have two options:
a) Download from [here](https://deepseeresources.s3.us-east-2.amazonaws.com/CelebA.zip) or b) predict them yourself. 
3. **Create dataset splits**: On the [author's website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), follow the link `Train/Val/Test Partitions` and download the file with dataset partitions `CelebA_list_eval_partition.txt`. It is located in the `Eval` folder. Update the paths in `data/celeba_partition.py` and run `python data/celeba_partition.py`.

### Custom Dataset
You need one folder containing the images and another folder with semantic masks. The filenames for the image name and label should be the same (ignoring extensions). For example, the image `0001.jpg` and the semantic mask `0001.png` would belong to the same sample. You can then copy and adapt one of the `*_dataset.py` classes to your new dataset. 



## Code Structure
- `train.py`: Training script. Run this via the bash scripts in `scripts/train/`.
- `demo.py`: This script provides the interface for loading data and running inference for the [demo notebook](demo.ipynb).
- `data/`: Dataset classes, pre-processing scripts and dataset preparation scripts.
- `deepsee_models/sr_model.py`: Interface to encoder, generator and discriminator.
- `deepsee_models/networks/`: Actual model code (residual block, normalization block, encoder, discriminator, etc.)
- `evaluator`: Evaluates during training and testing. `evaluator/evaluate_folder.py` computes scores for two folders, one containing the upscaled images, and on the ground truth.
- `managers/`: We have different manager classes for training, inference and demo.
- `options/`: Contains the configurations and command line parameters for training and testing.
- `scripts`: Contains scripts with pre-defined configurations.
- `util/`: Code for logging, visualizations and more.

## Citation
> Marcel Christoph Bühler, Andrés Romero, and Radu Timofte.<br> 
Deepsee: Deep disentangled semantic explorative extreme super-resolution.<br> 
In The 15th Asian Conference on Computer Vision (ACCV), 2020. 


[Bibtex](docs/citation.txt)


## License
Copyright belongs to the authors.
All rights reserved. Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

For the SPADE part: Copyright (C) 2019 NVIDIA Corporation.


## Acknowledgments
We built on top of the code from [SPADE](https://github.com/nvlabs/spade).
Thanks to Jiayuan Mao for [Synchronized Batch Normalization code](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch),
mseitzer for the [FID implementation in pytorch](https://github.com/mseitzer/pytorch-fid.git) 
and the authors of [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity).

