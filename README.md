# Awecome-Domain-Generalizable-Person-Re-identification
Domain Generalization for Person Re-identification: A Survey Towards Domain-Agnostic Person Matching

[![awesome](https://img.shields.io/badge/awesome-yes-critical?style=flat&logo=awesome-lists&labelColor=purple)](https://github.com/sindresorhus/awesome)
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=PerceptualAI-Lab/Awesome-Domain-Generalizable-Person-Re-ID)](https://github.com/PerceptualAI-Lab/Awesome-Domain-Generalizable-Person-Re-ID)
[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg)](https://arxiv.org/abs/2506.01061)
[![Stars](https://img.shields.io/github/stars/PerceptualAI-Lab/Awesome-Domain-Generalizable-Person-Re-ID.svg?style=social&label=Star)]([https://github.com/PerceptualAI-Lab/Awesome-Domain-Generalizable-Person-Re-ID])


This repository provides a curated collection of papers, benchmarks, and resources from our survey:  
**"Domain Generalization for Person Re-identification: A Survey Towards Domain-Agnostic Person Matching"** (Neurocomputing 2025).

> 📝 **Authors**: Hyeonseo Lee, Juhyun Park, Jihyong Oh, Chanho Eom†

> 🎓 **Institution**: Chung-Ang University, GSAIM  

---

## 📘 Abstract

Person Re-identification (ReID) aims to retrieve images of the same individual captured across non-overlapping camera views, making it a critical component of intelligent surveillance systems. Traditional ReID methods assume that the training and test domains share similar characteristics and primarily focus on learning discriminative features within a given domain. However, they often fail to generalize to unseen domains due to domain shifts caused by variations in viewpoint, background, and lighting conditions. To address this issue, Domain-Adaptive ReID (DA-ReID) methods have been proposed. These approaches incorporate unlabeled target domain data during training and improve performance by aligning feature distributions between source and target domains. However, their reliance on access to target domain data limits their scalability and makes them less suitable for real-world deployments, where such data may not be available in advance. Domain-Generalizable ReID (DG-ReID) tackles a more realistic and challenging setting by aiming to learn domain-invariant features without relying on any target domain data. Recent methods have explored various strategies to enhance generalization across diverse environments, but the field remains relatively underexplored. In this paper, we present a comprehensive survey of DG-ReID. We first review the architectural components of DG-ReID including the overall setting, commonly used backbone networks and multi-source input configurations. Then, we categorize and analyze domain generalization modules that explicitly aim to learn domain-invariant and identity-discriminative representations. To examine the broader applicability of these techniques, we further conduct a case study on a related task that also involves distribution shifts. Finally, we discuss recent trends, open challenges, and promising directions for future research in DG-ReID. To the best of our knowledge, this is the first systematic survey dedicated to DG-ReID.


---

## 📚 Contents

- [📣 News](#-news)
- [🔖 Citation](#-citation)
- [🔍 Survey Paper](#-survey-paper)
- [📄 Paper List](#-paper-list)
- [📊 Datasets & Benchmarks](#-datasets--benchmarks)
- [📈 Evaluation Metrics](#-evaluation-metrics)

---

## 📣 News

- 📌 2025-06: Paper released to ArXiv.
- 🚀 2025-05: Repository initialized.

---

## 🔖 Citation

If you find this survey helpful, please consider citing us:

```citation
@article{Kye2025AceVFI,
  title={AceVFI: A Comprehensive Survey of Advances in Video Frame Interpolation},
  author={Kye, Dahyeon and Roh, Changhyun and Ko, Sukhun and Eom, Chanho and Oh, Jihyong},
  journal={arXiv preprint arXiv:2506.01061},
  year={2025}
}
```
---

## 🧩 Community Contribution

We welcome contributions from the VFI research community!

If you have a new method, dataset, or related resource that fits within the scope of this VFI repository,
please feel free to submit a pull request (PR) with the following:

A brief description of your method/resource.

Relevant links (e.g., arXiv, project page, code).

Suggested placement (e.g., under “2.3 Diffusion Model-based", "6.4 Joint Task”).

Our maintainers will review your submission and merge it if appropriate.
We hope this page will grow into a collaborative hub for Video Frame Interpolation (VFI) research.

---

## 🔍 Survey Paper

You can find the preprint of our survey here:  
📄 [arXiv:2506.01061](https://arxiv.org/abs/2506.01061)


The overview of our survey paper:
![taxonomy](https://github.com/PerceptualAI-Lab/Awesome-Domain-Generalizable-Person-Re-ID/blob/main/figures/taxonomy.png)

---


## 📄 Paper List

We categorize recent VFI papers by methodology:  


## Normalization-based
<table>
<thead>
<tr>
<th align="left">Title</th>
<th align="center">Publication</th>
<th align="center">Date</th>
</tr>
</thead>
<tbody>
<tr><td align="left"><a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=44281">Fractional Frame Rate Up-conversion Using Weighted Median Filters</a></td><td align="center">IEEE Transactions on Consumer Electronics</td><td align="center">1989</td></tr>
<tr><td align="left"><a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Niklaus_Video_Frame_Interpolation_CVPR_2017_paper.pdf">Video Frame Interpolation via Adaptive Convolution</a></td><td align="center">CVPR</td><td align="center">2017</td></tr>
</tbody>
</table>


## Mixture-of-Experts-based
<table>
<thead>
<tr>
<th align="left">Title</th>
<th align="center">Publication</th>
<th align="center">Date</th>
</tr>
</thead>
<tbody>

</tbody>
</table>

## Memory-based
<table>
<thead>
<tr>
<th align="left">Title</th>
<th align="center">Publication</th>
<th align="center">Date</th>
</tr>
</thead>
<tbody>

</tbody>
</table>


## Meta-learning-based
<table>
<thead>
<tr>
<th align="left">Title</th>
<th align="center">Publication</th>
<th align="center">Date</th>
</tr>
</thead>
<tbody>

</tbody>
</table>


## Data-driven learning-based 
<table>
<thead>
<tr>
<th align="left">Title</th>
<th align="center">Publication</th>
<th align="center">Date</th>
</tr>
</thead>
<tbody>

</tbody>
</table>

## CLIP-based

<table>
<thead>
<tr>
<th align="left">Title</th>
<th align="center">Publication</th>
<th align="center">Date</th>
</tr>
</thead>
<tbody>

</tbody>
</table>


## Others
<table>
<thead>
<tr>
<th align="left">Title</th>
<th align="center">Publication</th>
<th align="center">Date</th>
</tr>
</thead>
<tbody>

</tbody>
</table>

## Datasets

<table>
<thead>
<tr>
<th align="left">Title</th>
<th align="center">Publication</th>
<th align="center">Date</th>
</tr>
</thead>
<tbody>
<tr><td align="left"><a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/333f81766b242b1837fa65c2172afb76-Paper-Datasets_and_Benchmarks_Track.pdf">LAVIB: A Large-scale Video Interpolation Benchmark</a></td><td align="center">NeurIPS</td><td align="center">2024</td></tr>
<tr><td align="left"><a href="https://arxiv.org/pdf/2407.02371">OPENVID-1M: A Large-scale High-quality Dataset for Text-to-video Generation</a></td><td align="center">ICLR</td><td align="center">2024</td></tr>
<tr><td align="left"><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Siyao_Deep_Animation_Video_Interpolation_in_the_Wild_CVPR_2021_paper.pdf">Deep animation video interpolation in the wild</a></td><td align="center">CVPR</td><td align="center">2021</td></tr>
<tr><td align="left"><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Bain_Frozen_in_Time_A_Joint_Video_and_Image_Encoder_for_ICCV_2021_paper.pdf">Frozen in Time: A Joint Video and Image Encoder for End-to-end Retrieval</a></td><td align="center">ICCV</td><td align="center">2021</td></tr>
<tr><td align="left"><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Sim_XVFI_eXtreme_Video_Frame_Interpolation_ICCV_2021_paper.pdf">XVFI: Extreme Video Frame Interpolation</a></td><td align="center">ICCV</td><td align="center">2021</td></tr>
<tr><td align="left"><a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Niklaus_Softmax_Splatting_for_Video_Frame_Interpolation_CVPR_2020_paper.pdf">Softmax Splatting for Video Frame Interpolation</a></td><td align="center">CVPR</td><td align="center">2020</td></tr>
<tr><td align="left"><a href="https://cdn.aaai.org/ojs/6693/6693-13-9922-1-10-20200521.pdf">Channel Attention Is All You Need for Video Frame Interpolation</a></td><td align="center">AAAI</td><td align="center">2020</td></tr>
<tr><td align="left"><a href="https://openreview.net/pdf/3ae72db22e8443112a8e7e61a943c8044053e135.pdf">MEMC-Net: Motion Estimation and Motion Compensation Driven Neural Network for Video Interpolation and Enhancement</a></td><td align="center">IEEE Transactions on Pattern Analysis and Machine Intelligence</td><td align="center">2019</td></tr>
<tr><td align="left"><a href="https://arxiv.org/pdf/1711.09078">Video enhancement with task-oriented flow</a></td><td align="center">International Journal of Computer Vision</td><td align="center">2019</td></tr>
<tr><td align="left"><a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Su_Deep_Video_Deblurring_CVPR_2017_paper.pdf">Deep Video Deblurring for Hand-held Cameras</a></td><td align="center">CVPR</td><td align="center">2017</td></tr>
<tr><td align="left"><a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf">Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring</a></td><td align="center">CVPR</td><td align="center">2017</td></tr>
<tr><td align="left"><a href="https://arxiv.org/pdf/1702.02463">Video Frame Synthesis Using Deep Voxel Flow</a></td><td align="center">ICCV</td><td align="center">2017</td></tr>
<tr><td align="left"><a href="https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.pdf">A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation</a></td><td align="center">CVPR</td><td align="center">2016</td></tr>
<tr><td align="left"><a href="https://arxiv.org/pdf/1212.0402">UCF101: A Dataset of 101 Human Actions Classes From Videos in the Wild</a></td><td align="center">CRCV</td><td align="center">2012</td></tr>
<tr><td align="left"><a href="https://www.cvlibs.net/publications/Geiger2012CVPR.pdf">Are We Ready for Autonomous Driving? the Kitti Vision Benchmark Suite</a></td><td align="center">CVPR</td><td align="center">2012</td></tr>
<tr><td align="left"><a href="https://files.is.tue.mpg.de/black/papers/ButlerECCV2012.pdf">A Naturalistic Open Source Movie for Optical Flow Evaluation</a></td><td align="center">ECCV</td><td align="center">2012</td></tr>
<tr><td align="left"><a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4408903">A Database and Evaluation Methodology for Optical Flow</a></td><td align="center">International Journal of Computer Vision</td><td align="center">2011</td></tr>
<tr><td align="left"><a href="https://media.xiph.org/video/derf/">Xiph.org Video Test Media (derf's Collection)</a></td><td align="center"></td><td align="center">1994</td></tr>
</tbody>
</table>

## 📊 Datasets & Benchmarks 

We include commonly used datasets for evaluating VFI performance.  
Datasets are categorized into **Triplet** and **Multi-frame** types depending on the supervision format.

### 🔹 Triplet Datasets

Early learning-based VFI approaches primarily rely on triplet datasets, where two input frames are used to predict the temporally centered GT frame.


| Dataset    | Venue   | Type | Resolution                | Split        | #Videos / #Triplets | URL                                           |
| ---------- | ------- | ---- | ------------------------- | ------------ | ------------------- | --------------------------------------------- |
| Middlebury  | IJCV'11   | 🔹 T | ≤ 640×480 (VGA)         | test         | 12                  | [🔗](https://vision.middlebury.edu/flow/data/) |
| UCF101     | CRCV'12 | 🔹 T  | 256×256                   | test         | 379                 | [🔗](https://www.crcv.ucf.edu/data/UCF101.php) |
| Vimeo90K   | IJCV'19 | 🔹 T  | 448×256                   | train / test | 51,312 / 3,782      | [🔗](http://toflow.csail.mit.edu/)             |
| SNU-FILM   | AAAI'20 | 🔹 T  | ≤ 1280×720 (HD)           | test         | 1,240               | [🔗](https://myungsub.github.io/CAIN/)         |
| ATD-12K    | CVPR'21 | 🔹 T  | 1280×720, 1920×1080 (FHD) | train / test | 10,000 / 2,000      | [🔗](https://github.com/lisiyao21/AnimeInterp) |

---
## 💫 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PerceptualAI-Lab/Awesome-Domain-Generalizable-Person-Re-ID&type=Date)](https://www.star-history.com/#PerceptualAI-Lab/Awesome-Domain-Generalizable-Person-Re-ID&Date)
