# Improving Diffusion-Based Generative Models via Approximated Optimal Transport \(EDM-AOT\)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-diffusion-based-generative-models/image-generation-on-cifar-10)](https://paperswithcode.com/sota/image-generation-on-cifar-10?p=improving-diffusion-based-generative-models)

![Teaser image](https://github.com/large-scale-kim/EDM-AOT/blob/main/docs/tangent_edm.png)
![Teaser image](https://github.com/large-scale-kim/EDM-AOT/blob/main/docs/tanget_aot.png)

**Improving Diffusion-Based Generative Models via Approximated Optimal Transport**<br>
Daegyu Kim,  Jooyoung Choi, Chaehun Shin, Uiwon Hwang, Sungroh Yoon

[arXiv](https://arxiv.org/abs/2403.05069)

**Abstract**
###### *We introduce the Approximated Optimal Transport (AOT) technique, a novel training scheme for diffusion-based generative models. Our approach aims to approximate and integrate optimal transport into the training process, significantly enhancing the ability of diffusion models to estimate the denoiser outputs accurately. This improvement leads to ODE trajectories of diffusion models with lower curvature and reduced truncation errors during sampling. We achieve superior image quality and reduced sampling steps by employing AOT in training. Specifically, we achieve FID scores of 1.88 with just 27 NFEs and 1.73 with 29 NFEs in unconditional and conditional generations, respectively. Furthermore, when applying AOT to train the discriminator for guidance, we establish new state-of-the-art FID scores of 1.68 and 1.58 for unconditional and conditional generations, respectively, each with 29 NFEs. This outcome demonstrates the effectiveness of AOT in enhancing the performance of diffusion models.*

This is implementation code of [Improving Diffusion-Based Generative Models via Approximated Optimal Transport](https://arxiv.org/abs/2403.05069).

This code is based on [EDM](https://github.com/NVlabs/edm).

# Settings

We use sample libraries of settings of [EDM](https://github.com/NVlabs/edm).

```.bash
conda env create -f environment.yml
conda activate edm
```
# Pre-trained model

You can download our pre-trained [unconditoinal](https://drive.google.com/file/d/1y-79-IKw15BaCHJRznC8fUKQ9lQR2I_M/view?usp=sharing) and [conditional](https://drive.google.com/file/d/1KOSnBal7Mf1wVLwOiKgOachxcwfsDmum/view?usp=sharing) models.

To generate images using our model, run [generate.py](https://github.com/large-scale-kim/EDM-AOT/blob/main/generate.py).
```.bash
# Hyper-parameter settings of Heun's sampler. rho = 90 and steps = 14.
torchrun --nproc_per_node=1 --standalone generate.py --network NETWORK_DIR --seeds 0-49999 --outdir OUTPUT_DIR --subdirs \
     --batch 200  --rho 90 --steps 14
```

|Hyper-parameters, uncond| Options | NFE $\downarrow$ | FID $\downarrow$|
|-|-|-|-|
|$\rho$ = 7, steps = 18 \(the same as [EDM](https://github.com/NVlabs/edm)\) |```--rho 7 --steps 18 ``` | 35| 1.95 |
|$\rho$ = 90, steps = 14  |``` --rho 90 --steps 14 ```| 27|**1.88** |

|Hyper-parameters, cond| Options | NFE $\downarrow$ | FID $\downarrow$|
|-|-|-|-|
|$\rho$ = 7, steps = 18 \(the same as [EDM](https://github.com/NVlabs/edm)\) |```--rho 7 --steps 18 ``` | 35| 1.79 |
|$\rho$ = 72, steps = 15  |``` --rho 72 --steps 15 ```| 29|**1.73** |

# Training and Evaluation

You can train EDM-based diffuion models with our AOT using [train.py](https://github.com/large-scale-kim/EDM-AOT/blob/main/train.py).

We edit [loss.py](https://github.com/large-scale-kim/EDM-AOT/blob/main/training/loss.py) and [training_loop.py](https://github.com/large-scale-kim/EDM-AOT/blob/main/training/training_loop.py) for our AOT techniques.

For uncondtional models, 
```.bash
torchrun --standalone --nproc_per_node=4 train.py --outdir OUTPUT_DIR  --data DATASET_DIR  --cond 0 --arch ncsnpp \
         --batch 256  --aot 512
```
For condtional models, 
```.bash
torchrun --standalone --nproc_per_node=4 train.py --outdir OUTPUT_DIR  --data DATASET_DIR  --cond 1 --arch ncsnpp \
         --batch 256  --aot 2048
```

To measure the FID score of sampled images, run [fid.py](https://github.com/large-scale-kim/EDM-AOT/blob/main/fid.py).
```.bash
python fid.py calc --ref DATASET_npz --images OUTPUT_DIR
```
To prepare datasets for training and evaluation, refer 'Preparing datasets' section of [EDM](https://github.com/NVlabs/edm#preparing-datasets).

