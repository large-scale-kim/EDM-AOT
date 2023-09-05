# Improving Diffusion-Based Generative Models via Approximated Optimal Transport \(EDM-AOT\)

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
We edit [loss.py]() and [training_loop.py]() for AOT techniques.
```.bash
torchrun --standalone --nproc_per_node=4 train.py --outdir OUTPUT_DIR  --data DATASET  --cond 0 --arch ncsnpp \
        --batch-gpu 32 --batch 128  --large_batch 512
```

To measure the FID score of sampled images, run [fid.py](https://github.com/large-scale-kim/EDM-AOT/blob/main/fid.py).
```.bash
python fid.py calc --ref DATASET_npz --images OUTPUT_DIR
```
To prepare datasets for training and evaluation, refer [EDM](https://github.com/NVlabs/edm).

