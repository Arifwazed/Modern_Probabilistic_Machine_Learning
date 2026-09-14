# Modern_Probabilistic_Machine_Learning_Project
# DDPM vs Full-Covariance GMM on Two Moons

This project investigates whether the additional flexibility of iterative
Denoising Diffusion Probabilistic Models (DDPMs) provides sufficient benefit
to justify their computational cost compared with a full-covariance Gaussian
Mixture Model (GMM) for a low-dimensional, structured nonlinear distribution.

## Research Question

For a low-dimensional, structured nonlinear distribution such as Two Moons,
does the additional flexibility of iterative DDPM sampling provide enough
benefit to justify its computational cost compared with a full-covariance
Gaussian mixture model?

## Methods

Two probabilistic generative approaches are compared:

- Denoising Diffusion Probabilistic Model (DDPM)
- Full-covariance Gaussian Mixture Model (GMM)

Two DDPM configurations are evaluated:

- T = 100
- T = 1000

The GMM uses BIC-based model selection over:

- K = 2
- K = 4
- K = 8
- K = 16

## Dataset

The experiments use a synthetic Two Moons dataset containing 10,000 samples
with Gaussian noise (sigma = 0.08).

- Training set: 8,000 samples
- Held-out set: 2,000 samples
- StandardScaler fitted using the training set only

## Evaluation

The models are evaluated using:

- Mean nearest-neighbour distance to held-out data
- Sampling time
- Qualitative visualization of generated samples

## Repository Contents

`DDPM_vs_GMM_Two_Moons.ipynb` contains the complete experimental workflow,
including:

1. Data generation and preprocessing
2. DDPM implementation and training
3. DDPM sampling
4. GMM fitting and BIC-based model selection
5. Sample generation
6. Quantitative evaluation
7. Visualization of results

## Reproducibility

The notebook contains the implementation and experimental configuration
required to reproduce the reported results.

Install the required dependencies using:

```bash
pip install -r requirements.txt
