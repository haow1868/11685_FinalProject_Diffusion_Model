# 11685 Final Project – Diffusion Model

This repository contains the code for our 11-685 final project on diffusion models for image generation.  
We build on the provided `hw5_student_starter_code` to implement and experiment with:

- **DDPM** – Denoising Diffusion Probabilistic Model  
- **DDIM** – Denoising Diffusion Implicit Model (fast deterministic sampler)  
- **U-Net** denoiser  
- **VAE** – for latent diffusion (128×128 images → 32×32 latent space)  
- **Classifier-Free Guidance (CFG)** – class-conditional sampling  
- End-to-end **training and inference** for multiple configurations

---

## Repository Structure

- `hw5_student_starter_code/`  
  Core implementation of DDPM, DDIM, U-Net, VAE, CFG, and the training / sampling logic
  adapted from the course starter code.

- Colab notebooks (and related scripts)  
  Used to train and evaluate **six latent DDPM + CFG configurations** on ImageNet-100.

---

## Experimental Configurations

We train six main latent DDPM + CFG models that differ in data subset, noise schedule,
EMA usage, batch size, and learning rate:

1. **FULL-LINEAR-256** – 100 classes, linear schedule, no EMA, batch size 256, LR = 2e-4  
2. **SUBSET-LINEAR-512** – 10 classes, linear schedule, no EMA, batch size 512, LR = 3e-4  
3. **SUBSET-LINEAR-EMA** – 10 classes, linear schedule, with EMA, batch size 512, LR = 3e-4  
4. **SUBSET-COSINE-512** – 10 classes, cosine schedule, no EMA, batch size 512, LR = 3e-4  
5. **SUBSET-COSINE-EMA** – 10 classes, cosine schedule, with EMA, batch size 512, LR = 3e-4  
6. **SUBSET-COSINE-EMA-LOWLR-256** – 10 classes, cosine schedule, with EMA,
   batch size 256, LR = 1e-4  

These configurations are evaluated using **FID** and **Inception Score (IS)** in the report.

---

## How to Use

- To inspect or modify the model and training code, start from  
  `hw5_student_starter_code/` (e.g., `train.py`, model definitions, and configs).
- Training was primarily run in **Google Colab**; you can open the notebooks in this
  repository and point them to your ImageNet-100 data to reproduce our results.
- For command-line usage, please see the argument definitions in `train.py`
  and adjust dataset paths, hyperparameters, and configuration options as needed.

---

## Acknowledgements

The base implementation and starter code are provided as part of the
**11-685: Introduction to Deep Learning** course assignment on diffusion models.
