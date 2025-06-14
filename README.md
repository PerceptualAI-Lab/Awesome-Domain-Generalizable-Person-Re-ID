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

- 📌 2025-06: Paper accepted on Neurocomputing and repository initialized
- 🚀 2025-05: Repository initialized.

---

## 🔖 Citation

If you find this survey helpful, please consider citing us:

Comming Soon!

```citation

```
---

## 🧩 Community Contribution

We welcome contributions from the Person Re-identification (ReID) research community!

If you have a new method, dataset, or related resource that fits within the scope of this ReID repository,
please feel free to submit a pull request (PR) with the following:

A brief description of your method/resource.

Relevant links (e.g., arXiv, project page, code).

Our maintainers will review your submission and merge it if appropriate.
We hope this page will grow into a collaborative hub for ReID research.

---

## 🔍 Survey Paper

You can find the preprint of our survey here:  
📄 Comming Soon!


The overview of our survey paper:
![taxonomy](https://github.com/PerceptualAI-Lab/Awesome-Domain-Generalizable-Person-Re-ID/blob/main/figures/taxonomy.png)

---


## 📄 Paper List

We categorize recent DG-ReID papers by methodology:  


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
  <tr><td align="left"><a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=44281](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jin_Style_Normalization_and_Restitution_for_Generalizable_Person_Re-Identification_CVPR_2020_paper.pdf">Style Normalization and Restitution for Generalizable Person Re-Identification</a></td><td align="center">CVPR</td><td align="center">2020</td></tr>
  <tr><td align="left"><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_Meta_Batch-Instance_Normalization_for_Generalizable_Person_Re-Identification_CVPR_2021_paper.pdf">Meta Batch-Instance Normalization for Generalizable Person Re-Identification</a></td><td align="center">CVPR</td><td align="center">2021</td></tr>
  <tr><td align="left"><a href="https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740209.pdf">Adaptive Cross-Domain Learning for Generalizable Person Re-Identification</a></td><td align="center">ECCV</td><td align="center">2022</td></tr>
  <tr><td align="left"><a href="https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740279.pdf">Dynamically Transformed Instance Normalization Network for Generalizable Person Re-Identification</a></td><td align="center">ECCV</td><td align="center">2022</td></tr>
  <tr><td align="left"><a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08753.pdf">Rethinking Normalization Layers for Domain Generalizable Person Re-identification</a></td><td align="center">ECCV</td><td align="center">2022</td></tr>
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
  <tr><td align="left"><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_Generalizable_Person_Re-Identification_With_Relevance-Aware_Mixture_of_Experts_CVPR_2021_paper.pdf">Generalizable Person Re-identification with Relevance-aware Mixture of Experts</a></td><td align="center">CVPR</td><td align="center">2021</td></tr>
  <tr><td align="left"><a href="https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740362.pdf">Mimic Embedding via Adaptive Aggregation: Learning Generalizable Person Re-identification</a></td><td align="center">ECCV</td><td align="center">2022</td></tr>
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
  <tr><td align="left"><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Song_Generalizable_Person_Re-Identification_by_Domain-Invariant_Mapping_Network_CVPR_2019_paper.pdf">Generalizable Person Re-identification by Domain-Invariant Mapping Network</a></td><td align="center">CVPR</td><td align="center">2019</td></tr>
  <tr><td align="left"><a href="https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560443.pdf">Interpretable and Generalizable Person Re-identification with Query-adaptive Convolution and Temporal Lifting</a></td><td align="center">ECCV</td><td align="center">2020</td></tr>
  <tr><td align="left"><a href="https://openaccess.thecvf.com/content/ICCV2023/papers/Ni_Part-Aware_Transformer_for_Generalizable_Person_Re-identification_ICCV_2023_paper.pdf">Part-Aware Transformer for Generalizable Person Re-identification</a></td><td align="center">ICCV</td><td align="center">2023</td></tr>
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
  <tr><td align="left"><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Bai_Person30K_A_Dual-Meta_Generalization_Network_for_Person_Re-Identification_CVPR_2021_paper.pdf">Person30K: A Dual-Meta Generalization Network for Person Re-Identification</a></td><td align="center">CVPR</td><td align="center">2021</td></tr>
  <tr><td align="left"><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Learning_to_Generalize_Unseen_Domains_via_Memory-based_Multi-Source_Meta-Learning_for_CVPR_2021_paper.pdf">Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification</a></td><td align="center">CVPR</td><td align="center">2021</td></tr>
  <tr><td align="left"><a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Ni_Meta_Distribution_Alignment_for_Generalizable_Person_Re-Identification_CVPR_2022_paper.pdf">Meta distribution alignment for generalizable person re-identification</a></td><td align="center">CVPR</td><td align="center">2022</td></tr>
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
  <tr><td align="left"><a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Cloning_Outfits_From_Real-World_Images_to_3D_Characters_for_Generalizable_CVPR_2022_paper.pdf">Cloning Outfits From Real-World Images to 3D Characters for Generalizable Person Re-Identification</a></td><td align="center">CVPR</td><td align="center">2022</td></tr>
  <tr><td align="left"><a href="https://openaccess.thecvf.com/content/ICCV2023/papers/Dou_Identity-Seeking_Self-Supervised_Representation_Learning_for_Generalizable_Person_Re-Identification_ICCV_2023_paper.pdf">Identity-Seeking Self-Supervised Representation Learning for Generalizable Person Re-identification</a></td><td align="center">ICCV</td><td align="center">2023</td></tr>
  <tr><td align="left"><a href="https://arxiv.org/pdf/2411.11471">Generalizable Person Re-identification via Balancing Alignment and Uniformity</a></td><td align="center">NeurIPS</td><td align="center">2024</td></tr>
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
  <tr><td align="left"><a href="https://dl.acm.org/doi/full/10.1145/3701036">CLIP-DFGS: A Hard Sample Mining Method for CLIP in Generalizable Person Re-Identification</a></td><td align="center">ACM TOMM</td><td align="center">2024</td></tr>
  <tr><td align="left"><a href="https://ieeexplore.ieee.org/abstract/document/10858181">CILP-FGDI: Exploiting Vision-Language Model for Generalizable Person Re-Identification</a></td><td align="center">IEEE Transactions on Information Forensics and Security</td><td align="center">2025</td></tr>
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
  <tr><td align="left"><a href="https://openaccess.thecvf.com/content/WACV2024/papers/Li_Mitigate_Domain_Shift_by_Primary-Auxiliary_Objectives_Association_for_Generalizing_Person_WACV_2024_paper.pdf">Mitigate Domain Shift by Primary-Auxiliary Objectives Association for Generalizing Person ReID</a></td><td align="center">WACV</td><td align="center">2024</td></tr>
  <tr><td align="left"><a href="https://ojs.aaai.org/index.php/AAAI/article/view/28468">Diversity-Authenticity Co-constrained Stylization for Federated Domain Generalization in Person Re-identification</a></td><td align="center">AAAI</td><td align="center">2024</td></tr>
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
  <tr><td align="left"><a href=""> </a></td><td align="center">CVPR</td><td align="center">2020</td></tr>
  <tr><td align="left"><a href=""> </a></td><td align="center">CVPR</td><td align="center">2020</td></tr>
  <tr><td align="left"><a href=""> </a></td><td align="center">CVPR</td><td align="center">2020</td></tr>
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
