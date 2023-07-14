# FigGen @ICLR 2023

## [FigGen: Text to Scientific Figure Generation](https://arxiv.org/abs/2306.00800)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2306.00800)

[Juan A. Rodríguez](https://scholar.google.es/citations?user=0selhb4AAAAJ&hl=en), [David Vázquez](https://scholar.google.es/citations?user=1jHvtfsAAAAJ&hl=en), [Issam Laradji](https://scholar.google.ca/citations?user=8vRS7F0AAAAJ&hl=en), [Marco Pedersoli](https://scholar.google.com/citations?user=aVfyPAoAAAAJ&hl=en), [Pau Rodríguez](https://scholar.google.com/citations?user=IwBx73wAAAAJ)

-----------
[ServiceNow Research, Montréal, Canada](https://www.servicenow.com/research/)
[ÉTS Montreal, University of Québec](https://www.etsmtl.ca/)

------------------
OCR-VQGAN is an image encoder designed to generate images that display clear and readable text. We propose to add an **OCR perceptual loss** term to the overall VQGAN loss, that encourages the learned discrete latent space to encode text patterns (i.e. learn rich latent representations to decode clear text-within-images). 

We experiment with OCR-VQGAN in and a novel dataset of images of figures and diagrams from research papers, called [**Paper2Fig100k dataset**](https://zenodo.org/record/7299423#.Y2lzonbMKUl). We find that using OCR-VQGAN to encode images in Paper2Fig100k results in much better figure reconstructions.

This code is adapted from **VQGAN** at [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers), and [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion). The OCR detector model used in OCR Perceptual loss is the **CRAFT** model from [clovaai/CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch).

<!-- 
<p align="center">
  <a href="https://arxiv.org/abs/2306.00800"><img src="assets/ocr_v2.png" alt="comparison" width="600" border="0"></a>
</p> -->

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

## Related work

**[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) by Rombach et al, CVPR 2022 Oral.**

**[OCR-VQGAN: Taming Text-within-Image Generation](https://arxiv.org/abs/2210.11248) by Rodriguez et al, WACV 2023.**

-----------------

## Citation
If you use this code please cite the following paper:
```bibtex
@inproceedings{rodriguez2023ocr,
  title={OCR-VQGAN: Taming Text-within-Image Generation},
  author={Rodriguez, Juan A and Vazquez, David and Laradji, Issam and Pedersoli, Marco and Rodriguez, Pau},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3689--3698},
  year={2023}
}
```

## Contact
Juan A. Rodríguez (joanrg.ai@gmail.com). 
