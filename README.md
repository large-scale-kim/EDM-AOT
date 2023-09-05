# Improving Diffusion-Based Generative Models via Approximated Optimal Transport \(EDM-AOT\)
![Teaser image](./docs/tangent_edm.jpg)
![Teaser image](./docs/tangent_aot.jpg)
### Improving Diffusion-Based Generative Models via Approximated Optimal Transport

###### Daegyu Kim, Uiwon Hwang, Jooyoung Choi, Chaehun Shin, Sungroh Yoon
[arXiv]()

Abstract
###### *Recent studies have explored the application of Optimal Transport to ODE-based image generative models. These approaches enhance generative models by identifying the Optimal Transport connections between image and noise distributions. However, applying these approaches to diffusion-based generative models introduces computational efficiency issues, making their application challenging. Therefore, simulation-free models, such as Flow Matching, are the only viable option for implementing these approaches. To tackle this limitation, we propose the Approximated Optimal Transport (AOT) for diffusion-based generative models, which effectively estimates the Optimal Transport process. We make two primary contributions. Firstly, we illustrate the viability of obtaining AOT through mini-batch coupling. Secondly, we enhance the training of diffusion models by incorporating AOT, resulting in improved model performance. Notably, our method achieves state-of-the-art results in unconditional image generation on the CIFAR-10 dataset without requiring guidance models, obtaining a FID score of 1.88 with 27 NFE (the number of function evaluations). Additionally, we extend the application of AOT to the training of discriminator guidance models, leading to a new benchmark for unconditional image generation on the CIFAR-10 dataset utilizing guidance models with a remarkable FID score of 1.67 with 29 NFE.*

This is implementation code of [Improving Diffusion-Based Generative Models via Approximated Optimal Transport]().

This code is based on [EDM](https://github.com/NVlabs/edm).

# Settings

We use sample libraries of settings of [EDM](https://github.com/NVlabs/edm).

```.bash
conda env create -f environment.yml
conda activate edm
```
# Pre-trained model

You can download our pre-trained model in [this link](https://drive.google.com/file/d/1y-79-IKw15BaCHJRznC8fUKQ9lQR2I_M/view?usp=sharing).

To generate images using our model, run [generate.py](https://github.com/large-scale-kim/EDM-AOT/blob/main/generate.py).
```.bash
# Hyper-parameter settings of Heun's sampler. rho = 90 and steps = 14.
torchrun --nproc_per_node=1 --standalone generate.py --network NDEWORK_DIR --seeds 0-49999 --outdir OUTPUT_DIR --subdirs \
     --batch 200  --rho 90 --steps 14
```

|Hyper-parameters| Options | NFE $\downarrow$ | FID $\downarrow$|
|-|-|-|-|
|$\rho$ = 7, steps = 18 \(the same as [EDM](https://github.com/NVlabs/edm)\) |```--rho 7 --steps 18 ``` | 35| 1.95 |
|$\rho$ = 90, steps = 14  |``` --rho 90 --steps 14 ```| 27|**1.88** |

# Training and Evaluation

You can train EDM-based diffuion models with our AOT using [train.py](https://github.com/large-scale-kim/EDM-AOT/blob/main/train.py).

We edit [loss.py](https://github.com/large-scale-kim/EDM-AOT/blob/main/training/loss.py) and [training_loop.py](https://github.com/large-scale-kim/EDM-AOT/blob/main/training/training_loop.py) for our AOT techniques.
```.bash
torchrun --standalone --nproc_per_node=4 train.py --outdir OUTPUT_DIR  --data DATASET  --cond 0 --arch ncsnpp \
        --batch-gpu 32 --batch 128  --large_batch 512
```

To measure the FID score of sampled images, run [fid.py](https://github.com/large-scale-kim/EDM-AOT/blob/main/fid.py).
```.bash
python fid.py calc --ref DATASET_npz --images OUTPUT_DIR
```
To prepare datasets for training and evaluation, refer 'Preparing datasets' section of [EDM](https://github.com/NVlabs/edm#preparing-datasets).

