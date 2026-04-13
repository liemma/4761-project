# Project Proposal

**Emma Li and Irene Wu**

## Working Title

Biologically Informed Spatial Markov Random Fields for Transcriptome Data

## Motivation: Spatial Structure in Transcriptome Data

Gene expression is not a collection of independent measurements. Across tissue regions, expression profiles exhibit strong and biologically meaningful dependency: nearby regions often share regulatory environments, and functionally linked regions may display coordinated transcriptional activity. Many statistical frameworks ignore this structure and instead treat each region as independent, discarding information that is directly encoded in the data.

To improve power and interpretability, Markov Random Fields (MRFs) provide a principled way to model this kind of dependency. By defining a graph over spatial locations and encoding pairwise interactions between neighboring nodes, MRFs allow latent expression states to borrow strength from their neighbors. Prior work, including Lin et al. (2015), showed that MRF-based approaches can outperform naive independent models in recovering biologically meaningful expression patterns.

## Limitation of a Uniform Spatial Parameter

A key limitation of the Lin et al. framework is its use of a single scalar spatial coupling coefficient applied uniformly across all region pairs. This implies that all spatial relationships are equally strong, which is biologically implausible. Brain regions differ substantially in structural connectivity, functional co-activation, and transcriptional co-regulation.

We propose to replace this uniform spatial parameter with a weighted spatial graph \(W\), where each entry \(W_{bb'}\) reflects a biologically informed connectivity measure between regions \(b\) and \(b'\). These weights may be derived from an existing neuroanatomical atlas, functional connectivity resource, or coexpression reference. This preserves the MRF framework while making the spatial smoothing step more realistic and interpretable.

## Proposed Method

We focus on a **spatial-only MRF refinement**.

Let each brain region be a node in a graph, and let edges be weighted by biological connectivity. The latent state for each region is encouraged to agree with the states of strongly connected neighboring regions, with the degree of encouragement determined by the corresponding edge weights.

Relative to the original model, the main methodological change is:

- replacing a single global spatial coupling parameter with a weighted connectivity matrix
- preserving the MRF structure for spatial smoothing
- evaluating whether weighted spatial edges improve the biological plausibility of inferred expression states

## Current Scope Clarification

An earlier project direction considered separating spatial and temporal dependence by pairing a spatial MRF with a temporal Hidden Markov Model (HMM). That is **not** part of the current project scope.

For this project, we are focusing only on the spatial MRF component. Temporal modeling may be discussed as future work, but it is not part of the implementation or evaluation plan.

## Why This Matters

This refinement is worthwhile even on its own. A weighted spatial MRF:

- incorporates known biology directly into the prior structure
- may improve estimation by strengthening plausible region-to-region influence
- yields a more interpretable model than uniform spatial smoothing
- provides a practical and focused extension of the original framework

## Generalizability

Although we motivate the work using brain-region transcriptome data, the same idea applies more broadly. Any setting with region-resolved expression data can in principle use a weighted spatial graph, including spatial transcriptomics experiments in developmental biology, cancer, or tissue organization studies. The graph would simply be defined using the relevant notion of spatial or biological neighborhood.

## Planned Evaluation

We aim to:

1. reproduce a baseline spatial MRF setup inspired by prior work
2. construct a biologically informed region-to-region weight matrix
3. compare the weighted model against a uniform spatial baseline
4. assess differences in inferred expression patterns and interpretability

## Expected Contribution

The project delivers a computationally tractable, biologically grounded refinement of a spatial MRF framework for transcriptome analysis. Its main contribution is to replace uniform spatial smoothing with connectivity-aware smoothing, making the model better aligned with known regional relationships while keeping the method focused and feasible.

## Future Work

Possible future extensions include adding temporal structure, integrating multi-omic priors, or adapting the method to modern spatial transcriptomics platforms. However, these are outside the scope of the current project.
