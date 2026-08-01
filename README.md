# <div align="center">
# DISCtomo: A Python package for direct smoothness constrained seismic traveltime tomography and uncertainty analysis
**Chao Li**<sup>1</sup> &nbsp;&nbsp;
**Yangkang Chen**<sup>1</sup>

<sup>1</sup> Bureau of Economic Geology, John A. and Katherine G. Jackson School of Geosciences, The University of Texas at Austin  

![Stars](https://img.shields.io/github/stars/ChaoDeepGeo/DISC?style=social)
![Forks](https://img.shields.io/github/forks/ChaoDeepGeo/DISC?style=social)
![Repo Size](https://img.shields.io/github/repo-size/ChaoDeepGeo/DISC)
![Last Commit](https://img.shields.io/github/last-commit/ChaoDeepGeo/DISC)
![Top Language](https://img.shields.io/github/languages/top/ChaoDeepGeo/DISC)

</div>

---

## Abstract
We develop a open-source Python library DISCtomo, a flexible, compact, and user-friendly package for tomography based velocity building and uncertainty evaluation. Different from classic ray-tracing- or Eikonal equation-based tomography, DISCtomo directly constructs the tomography ray-path matrix from a precomputed traveltime table. In addition, shaping regularization is employed to control model smoothness, while the conjugate gradient (CG) algorithm is used to further refine the inversion results. Specifically, DISCtomo includes ray path matrix (RPM) construction functions for both 2D and 3D scenarios, Stein variational gradient descent (SVGD) for tomography uncertainty evaluation, and provides implementations of 2D and 3D models. Users can reproduce our examples or solve their own problems by just defining the geometry and initial model. Numerical experiments shown in the manuscript validate that DISCtomo is a effective and scalable package, which can be used for tomography at different scales in practice. 

---

## Framework

<p align="center">
  <img src="Figure/sketch.png" width="600"/>
</p>


<p align="center">
  <b>Example 1.</b>
A schematic illustration of DISC-based ray tracing from a receiver location to the source position.
The figure can be reproduced using the
<a href="https://github.com/ChaoDeepGeo/DISCtomo/blob/main/notebook/main_raytracing.ipynb">
ray-tracing notebook
</a>.
</p>

The workflow of DISC consists of the following main steps:

1. Conduct ray tracing based on a traveltime table estimated by Eikonal equation.
2. Construct ray path matrix based on he initial model
3. Compute the misfit between synthetic data and observed data.
4. Estimate the model update using CG iteration with shaping regularization and update the current model
5. Repeat steps 1-4 until convergence.

---

## Documents demonstration

- The data folder contains necessary models and dataset used to reproduce the numerical results.
- The Notebook folder contains necessary code to reproduce the tomography experiments with different geometries (e.g., crosswell and surface) for both active and passive source examples, including checkerboard, saltbody, 3D anormal and SVGD DISC for uncertain evaluation.
- The disc folder constains all the necessary functions for DISC package.
- DISC depends on the following external packages: 1. [Pyekfmm](https://github.com/aaspip/pyekfmm) 2. [Pyseistr](https://github.com/aaspip/pyseistr)

---
## Result

<p align="center">
  <img src="Figure/timecomp.png" width="850"/>
</p>

<p align="center">
  <b>Example 2.</b> (a) Real velocity model, (b) Ray tracing using DISC method, (c) Observed traveltime calculated using Eikonal equation, (d) Estimated traveltime by DISC method, and (e) The traveltime difference bewteen Figures 2(c) and 2(d). The figure can be reproduced using the
<a href="https://github.com/ChaoDeepGeo/DISCtomo/blob/main/notebook/main_traveltime_comp.ipynb">
traveltime_comparison notebook
</a>.
</p>

<p align="center">
  <img src="Figure/checkinv.png" width="850"/>
</p>

<p align="center">
  <b>Figure 3.</b> (a) Real checkerboard velocity model, (b) Initial constant velocity model, and (c) Inverted velocity model using DISC method.
</p>

<p align="center">
  <img src="Figure/saltinv.png" width="850"/>
</p>

<p align="center">
  <b>Figure 4.</b> (a) Real saltbody model, (b) Initial saltbody model, and (c) Inversion result of DISC.
</p>

<p align="center">
  <img src="Figure/syn3dini.png" width="850"/>
</p>

<p align="center">
  <b>Figure 5.</b> (a) Real 3D velocity model, (b) Initial model used for DISC inversion, and (c) Ray paths estimated by DISC method.
</p>

<p align="center">
  <img src="Figure/syn3dinv.png" width="850"/>
</p>

<p align="center">
  <b>Figure 6.</b> (a) Real 3D velocity model (ground truth) and (b) Inverted 3D result using DISC method. 
</p>

<p align="center">
  <img src="Figure/checkpar.png" width="850"/>
</p>

<p align="center">
  <b>Figure 7.</b> Generated initial particles used for ASVGD DISC inverison. The geometry is same as that shown in Figure 3.
</p>


<p align="center">
  <img src="Figure/checkasvgd.png" width="850"/>
</p>

<p align="center">
  <b>Figure 8.</b> (a) Real checkerboard model, (b) Estimated Vp mean value distribution using ASVGD DISC, and (c) Estimated Vp std map (uncertainty distribution). Higher std value means more uncertainties.
</p>


---



## Installation

To set up the environment, run:

```bash
conda env create -f environment.yml
conda activate your_env_name
```

---

## Reference 

> Li and Chen, 2026, Seismic Traveltime Tomography Using Direct Smoothness-Constrained Iterative Inversion, IEEE Transactions on Geoscience and Remote Sensing, 64, pp. 5909912.

### BibTex

```bibtex

@ARTICLE{DISC2026,
  author={Li, Chao and Chen, Yangkang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Seismic Traveltime Tomography Using Direct Smoothness-Constrained Iterative Inversion}, 
  year={2026},
  volume={64},
  number={},
  pages={5909912-5909912},
  doi={10.1109/TGRS.2026.3688440}
}


```





