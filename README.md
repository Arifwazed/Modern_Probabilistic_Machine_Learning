# Implementation and Empirical Evaluation of DDPM vs. GMM on Two Moons

This repository contains the implementation and experimental materials for a CSE756 Modern Probabilistic Machine Learning project comparing a Denoising Diffusion Probabilistic Model (DDPM) with a full-covariance Gaussian Mixture Model (GMM) on the Two Moons distribution.

## Research Question

> For a low-dimensional, structured nonlinear distribution such as Two Moons, does the additional flexibility of iterative DDPM sampling provide enough benefit to justify its computational cost compared with a full-covariance Gaussian mixture model?

## Hypotheses

- **H1:** Increasing the DDPM diffusion horizon from **T = 100** to **T = 1000** improves sample quality but increases sampling cost.
- **H2:** For the low-dimensional Two Moons distribution, a full-covariance GMM provides a better accuracy-efficiency trade-off than iterative DDPM sampling.

## Methods

### DDPM

Two independently trained DDPM configurations are evaluated:

- **T = 100**
- **T = 1000**

The noise-prediction network uses a 3-dimensional input consisting of the two noisy data coordinates and a normalized timestep, with three hidden layers of 128 ReLU units and a 2-dimensional noise output.

Training configuration:

- 70 epochs
- Adam optimizer
- Learning rate: 0.001
- Batch size: 512
- 60,000 generated noisy training pairs per DDPM
- Random seed: 42

The T = 100 schedule is adjusted so that its terminal signal/noise level is comparable with the T = 1000 reference schedule.

### Gaussian Mixture Model

Full-covariance GMMs are evaluated for:

- K = 2
- K = 4
- K = 8
- K = 16

Bayesian Information Criterion (BIC) is used for model selection. The final experiment selects **K = 16**.

## Dataset and Preprocessing

The experiment uses a synthetic Two Moons dataset:

- 10,000 total observations
- Gaussian noise: sigma = 0.08
- 80% training data
- 20% held-out data
- Stratified train/test split
- StandardScaler fitted on the training set only

## Evaluation

The generated samples are evaluated using:

1. Mean nearest-neighbour distance to the held-out data.
2. Sampling time.
3. Qualitative visualization of the generated distributions.

Generated sample sizes include **1,000** and **5,000** samples.

## Repository Contents

```text
Modern_Probabilistic_Machine_Learning/
│
├── README.md
├── CSE756_Project_Final_Organized.ipynb
├── requirements.txt
│
└── results/
    └── figures/
```

The notebook contains the complete experimental workflow, including:

1. Environment setup and reproducibility configuration
2. Two Moons data generation and preprocessing
3. DDPM diffusion schedules
4. DDPM model definition
5. DDPM training for T = 100 and T = 1000
6. DDPM reverse-diffusion sampling
7. Full-covariance GMM fitting
8. BIC-based GMM model selection
9. Quantitative evaluation
10. Runtime comparison
11. Final visualizations

## Reproducibility

The notebook is intended to be executed from beginning to end.

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

Then open:

```text
CSE756_Project_Final_Organized.ipynb
```

and run all cells in order.

The main experimental seed is **42**. DDPM training also uses deterministic seed offsets based on the diffusion horizon.

## Results Summary

The final verified experiment found that:

- Increasing DDPM from **T = 100** to **T = 1000** improves the mean nearest-neighbour result.
- The **full-covariance GMM with K = 16** achieves the lowest mean nearest-neighbour distance among the evaluated models.
- DDPM sampling requires substantially more computation because samples are generated through iterative reverse diffusion.
- On this low-dimensional Two Moons problem, the GMM provides a stronger accuracy-efficiency trade-off.

The quantitative results reported in the paper should be treated as the authoritative final results.

## Course

**CSE756 — Modern Probabilistic Machine Learning**

## Authors

Md Arif Wazed Hossain  
Nafisa Nawal 

## Project Repository

https://github.com/Arifwazed/Modern_Probabilistic_Machine_Learning
