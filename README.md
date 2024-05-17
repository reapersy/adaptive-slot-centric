<div align="center">

<!-- TITLE -->
# **Test-time Adaptation with Slot-Centric Models**

[![arXiv](https://img.shields.io/badge/cs.LG-arXiv:2203.11194-b31b1b.svg)](https://arxiv.org/abs/2203.11194)
[![Website](https://img.shields.io/badge/🌎-Website-blue.svg)](https://slot-tta.github.io/)
</div>

<img src="images/slot-tta_model_fig.png" alt="Model figure"/>


<!-- DESCRIPTION -->
## Abstract
This repository contains the source code for the experimental adaptation of a semi-supervised instance segmentation model, which seeks to improve its performance by adapting to out-of-distribution scenes. The model leverages slot-centric generative models in the process of adaptation. We provide detailed instructions on how to install and use the model, as well as code for conducting 2D RGB experiments.

## Code
### Installation
We use Pytorch 2.0 for all our experiments. We use [wandb](https://wandb.ai/) for logging the results.
Install conda (instructions [here](https://docs.conda.io/en/latest/miniconda.html)).
Create a conda environment using the below commands:
```
conda 