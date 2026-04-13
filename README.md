# 4761 Project

## Project Focus

This project studies **spatial structure in transcriptome data** using a biologically informed **spatial Markov Random Field (MRF)**. The current scope is intentionally narrower than the original brainstorming direction: we are **not** implementing the HMM-based temporal component anymore. Instead, we are focusing on improving the spatial side of the Lin et al. framework.

## Motivation

Gene expression measurements across brain regions are not independent. Nearby or functionally connected regions often share regulatory environments and exhibit correlated expression patterns. A spatial MRF provides a natural way to model this dependency by allowing latent expression states to borrow strength across related regions.

The original Lin et al. model used a single scalar spatial coupling parameter, which treats all region-to-region relationships as equally strong. That assumption is biologically unrealistic. Brain regions differ substantially in structural connectivity, functional co-activation, and transcriptional similarity.

## Core Idea

We replace the uniform spatial coupling term with a **connectivity-weighted spatial graph**.

Instead of assuming that every pair of neighboring brain regions interacts with the same strength, we use a weighted matrix \(W\) where each entry reflects a biologically informed measure of connectivity between regions. These weights can come from neuroanatomical atlases, coexpression resources, or other existing biological priors.

## Current Scope

The project is centered on:

- reproducing and understanding the spatial MRF component of the original framework
- replacing the uniform spatial parameter with a weighted spatial graph
- evaluating whether biologically informed spatial weights improve interpretability or performance
- keeping the implementation computationally practical and easy to analyze

Out of scope for the current version:

- HMM-based temporal modeling
- joint spatial-temporal factorization
- forward-backward temporal inference

## Why This Is Still Interesting

Even without the temporal HMM extension, the spatial refinement is meaningful on its own:

- it makes the prior more biologically realistic
- it tests whether domain knowledge improves smoothing and downstream inference
- it provides a cleaner and more tractable first milestone
- it can later serve as the foundation for broader spatiotemporal extensions

## Repository Plan

This repository will be used to organize:

- the project proposal and notes
- references to the original Lin et al. method
- code for spatial MRF experiments
- data-processing and evaluation scripts
- results and writeups

## Next Steps

1. Formalize the weighted spatial MRF model.
2. Identify the brain-region connectivity source used to define the weight matrix.
3. Build a baseline implementation of the spatial MRF.
4. Compare uniform versus weighted spatial coupling.
5. Summarize findings in figures and writeup.
