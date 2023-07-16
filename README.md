# FigGen @ICLR 2023

## [FigGen: Text to Scientific Figure Generation](https://arxiv.org/abs/2306.00800)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2306.00800)

[Juan A. Rodríguez](https://scholar.google.es/citations?user=0selhb4AAAAJ&hl=en), [David Vázquez](https://scholar.google.es/citations?user=1jHvtfsAAAAJ&hl=en), [Issam Laradji](https://scholar.google.ca/citations?user=8vRS7F0AAAAJ&hl=en), [Marco Pedersoli](https://scholar.google.com/citations?user=aVfyPAoAAAAJ&hl=en), [Pau Rodríguez](https://scholar.google.com/citations?user=IwBx73wAAAAJ)

[ServiceNow Research, Montréal, Canada](https://www.servicenow.com/research/)

[ÉTS Montreal, University of Québec](https://www.etsmtl.ca/)

------------------
**FigGen** is a latent diffusion model that generates scientific figures of papers conditioned on the text from the papers (text-to-figure). We use [OCR-VQGAN](https://github.com/joanrod/ocr-vqgan) to project scientific figures (images) into a latent representation, and use a latent diffusion model to learn a generator. We jointly train a Bert transformer to learn text embeddings and perform text-to-figure generation.

This code is adapted from **Latent Diffusion** at [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).


<p align="center">
  <a href="https://arxiv.org/abs/2306.00800"><img src="assets/qualitative3.png" alt="qualitative results" width="600" border="0"></a>
</p>

**Abstract**
>The generative modeling landscape has experienced tremendous growth in recent years, particularly in generating natural images and art. Recent techniques have shown impressive potential in creating complex visual compositions while delivering impressive realism and quality. However, state-of-the-art methods have been focusing on the narrow domain of natural images, while other distributions remain unexplored. In this paper, we introduce the problem of text-to-figure generation, that is creating scientific figures of papers from text descriptions. We present FigGen, a diffusion-based approach for text-to-figure as well as the main challenges of the proposed task. Code and models are available in this repository.

## Installation
Create a [conda](https://conda.io/) environment named `figgen`,
and activate it:

```bash
conda env create -f environment.yaml
conda activate figgen
pip install -e .
```

# Training

# Inference

# Results
<p align="center">
  <a href="https://arxiv.org/abs/2306.00800"><img src="assets/qualitative1.png" alt="qualitative results" width="600" border="0"></a>
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2306.00800"><img src="assets/qualitative2.png" alt="qualitative results" width="600" border="0"></a>
</p>

## Todo

- [ ] Automatically download Paper2Fig100k dataset (from Zenodo) and trained models (from HF) 


## Related work

**[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) by Rombach et al, CVPR 2022 Oral.**

**[OCR-VQGAN: Taming Text-within-Image Generation](https://arxiv.org/abs/2210.11248) by Rodriguez et al, WACV 2023.**

-----------------

## Citation
If you use this code please cite the following paper:
```bibtex
@article{rodriguez2023figgen,
  title={FigGen: Text to Scientific Figure Generation},
  author={Rodriguez, Juan A and Vazquez, David and Laradji, Issam and Pedersoli, Marco and Rodriguez, Pau},
  journal={arXiv preprint arXiv:2306.00800},
  year={2023}
}
```

## Contact
Juan A. Rodríguez (joanrg.ai@gmail.com). 
