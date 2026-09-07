# Verified 21-Paper Literature Review — Trustworthy Lumbar Spine MRI Diagnosis

**Compiled:** 2026-09-04 · **Method:** every citation checked against the Crossref DOI registry
(Google Scholar's underlying authoritative database). All are **journal papers** (no conferences),
matching your deck requirement. Replaces the fabricated placeholders in the Lit1–Lit3 slides.

> ⚠️ **Before submitting:** the "Key Finding" cells below are honest qualitative summaries.
> Skim each abstract once (links via DOI) to confirm the phrasing matches what you want to claim.
> No numbers were invented — where a paper reports specific figures, add them only after reading the abstract.

---

## Cluster 1 — Deep Learning for Spinal Diagnosis (Papers 1–8)

**1.** A. Jamaludin, T. Kadir, and A. Zisserman, "SpineNet: Automated classification and evidence visualization in spinal MRIs," *Med. Image Anal.*, vol. 41, pp. 63–73, 2017. doi: [10.1016/j.media.2017.07.002](https://doi.org/10.1016/j.media.2017.07.002)

**2.** A. Jamaludin, M. Lootus, T. Kadir, A. Zisserman, J. Urban, et al., "ISSLS Prize in Bioengineering Science 2017: Automation of reading of radiological features from magnetic resonance images (MRIs) of the lumbar spine without human intervention is comparable with an expert radiologist," *Eur. Spine J.*, vol. 26, no. 5, pp. 1374–1383, 2017. doi: [10.1007/s00586-017-4956-3](https://doi.org/10.1007/s00586-017-4956-3)

**3.** A. S. Al-Kafri, S. Sudirman, A. Hussain, D. Al-Jumeily, F. Natalia, H. Meidia, et al., "Boundary delineation of MRI images for lumbar spinal stenosis detection through semantic segmentation using deep neural networks," *IEEE Access*, vol. 7, pp. 43487–43501, 2019. doi: [10.1109/ACCESS.2019.2908002](https://doi.org/10.1109/ACCESS.2019.2908002)

**4.** F. Galbusera, G. Casaroli, and T. Bassani, "Artificial intelligence and machine learning in spine research," *JOR Spine*, vol. 2, no. 1, Art. no. e1044, 2019. doi: [10.1002/jsp2.1044](https://doi.org/10.1002/jsp2.1044)

**5.** W. Liawrungrueang, P. Kim, V. Kotheeranurak, K. Jitpakdee, P. Sarasombath, et al., "Automatic detection, classification, and grading of lumbar intervertebral disc degeneration using an artificial neural network model," *Diagnostics*, vol. 13, no. 4, Art. no. 663, 2023. doi: [10.3390/diagnostics13040663](https://doi.org/10.3390/diagnostics13040663)

**6.** G. Liu, L. Wang, S.-N. You, Z. Wang, S. Zhu, C. Chen, et al., "Automatic detection and classification of Modic changes in MRI images using deep learning: Intelligent assisted diagnosis system," *Orthop. Surg.*, vol. 16, no. 1, pp. 196–206, 2024. doi: [10.1111/os.13894](https://doi.org/10.1111/os.13894)

**7.** I. Ram, S. Kumar, and A. K. Keshri, "Classification of intervertebral disc using novel multi-branch convolutional residual network model," *Biomed. Signal Process. Control*, vol. 102, Art. no. 107332, 2025. doi: [10.1016/j.bspc.2024.107332](https://doi.org/10.1016/j.bspc.2024.107332)

**8.** Z. Karakuzu Güngör, H. Vehbi, and A. Cansın, "Automated detection of lumbar disc herniation at L4–L5 and L5–S1 levels on sagittal MRI using a YOLO-based deep learning model," *BMC Musculoskelet. Disord.*, vol. 27, Art. no. 715, 2026. doi: [10.1186/s12891-026-10124-4](https://doi.org/10.1186/s12891-026-10124-4)

| No. | Year | Author(s) | Focus | Technique(s) | Key Finding | Gap |
|---|---|---|---|---|---|---|
| 1 | 2017 | Jamaludin et al. | Classification + evidence visualization in spinal MRI | CNN (SpineNet) | Automated disc/pathology classification with localized evidence | Single-task, no calibration |
| 2 | 2017 | Jamaludin et al. | Automated reading of MRI features vs expert | Deep CNN | Reading comparable with expert radiologist | Heavy model, no uncertainty |
| 3 | 2019 | Al-Kafri et al. | Lumbar spinal stenosis detection | Semantic segmentation (DNN) | Automatic canal boundary delineation for stenosis | Segmentation only, no grading |
| 4 | 2019 | Galbusera et al. | Survey of AI/ML in spine research | Review | Broad applicability, validation gaps remain | No unified trustworthy framework |
| 5 | 2023 | Liawrungrueang et al. | IVD degeneration grading | Artificial neural network | Automated Pfirrmann-style grading | Single task, no calibration |
| 6 | 2024 | Liu et al. | Modic change detection | Deep learning classifier | Automated Modic change identification | Binary classes, no uncertainty |
| 7 | 2025 | Ram et al. | Intervertebral disc classification | Multi-branch residual CNN | Improved IVD classification | No MTL / calibration / XAI |
| 8 | 2026 | Karakuzu Güngör et al. | Disc herniation detection (L4–L5, L5–S1) | YOLO-based detector | Automated herniation detection on sagittal MRI | Detection only, no trust metrics |

---

## Cluster 2 — Multi-Task Learning in Medical Imaging (Papers 9–13)

**9.** S. Graham, Q. D. Vu, M. Jahanifar, S. E. A. Raza, F. Minhas, D. Snead, et al., "One model is all you need: Multi-task learning enables simultaneous histology image segmentation and classification," *Med. Image Anal.*, vol. 83, Art. no. 102685, 2023. doi: [10.1016/j.media.2022.102685](https://doi.org/10.1016/j.media.2022.102685)

**10.** Y. Zhou, H. Chen, Y. Li, Q. Liu, X. Xu, S. Wang, et al., "Multi-task learning for segmentation and classification of tumors in 3D automated breast ultrasound images," *Med. Image Anal.*, vol. 70, Art. no. 101918, 2021. doi: [10.1016/j.media.2020.101918](https://doi.org/10.1016/j.media.2020.101918)

**11.** F. Gao, H. Yoon, T. Wu, and X. Chu, "A feature transfer enabled multi-task deep learning model on medical imaging," *Expert Syst. Appl.*, vol. 143, Art. no. 112957, 2020. doi: [10.1016/j.eswa.2019.112957](https://doi.org/10.1016/j.eswa.2019.112957)

**12.** M. Xu, K. Huang, and X. Qi, "A regional-attentive multi-task learning framework for breast ultrasound image segmentation and classification," *IEEE Access*, vol. 11, pp. 5377–5392, 2023. doi: [10.1109/ACCESS.2023.3236693](https://doi.org/10.1109/ACCESS.2023.3236693)

**13.** X. Wu and G. Gou, "Uncertainty bidirectional guidance of multi-task mamba network for medical image classification and segmentation," *Signal Image Video Process.*, vol. 19, Art. no. 29, 2025. doi: [10.1007/s11760-024-03633-z](https://doi.org/10.1007/s11760-024-03633-z)

| No. | Year | Author(s) | Focus | Technique(s) | Key Finding | Gap |
|---|---|---|---|---|---|---|
| 9 | 2023 | Graham et al. | Histology segmentation + classification | Multi-task learning | One model handles both tasks | Histology, not spine |
| 10 | 2021 | Zhou et al. | Breast US segmentation + classification | Multi-task learning | MTL improves both tasks | Breast, not spine |
| 11 | 2020 | Gao et al. | Feature-transfer MTL on medical images | Feature transfer + MTL | Effective shared-feature learning | No trust metrics |
| 12 | 2023 | Xu et al. | Breast US segmentation + classification | Regional-attentive MTL | Attention improves multi-task performance | Not spine, no uncertainty |
| 13 | 2025 | Wu & Gou | Multi-task classification + segmentation | Multi-task Mamba + uncertainty guidance | Uncertainty improves multi-task learning | Not spine, no calibration |

---

## Cluster 3 — Uncertainty & Calibration (Papers 14–17)

**14.** A. Mehrtash, W. M. Wells, C. M. Tempany, P. Abolmaesumi, and T. Kapur, "Confidence calibration and predictive uncertainty estimation for deep medical image segmentation," *IEEE Trans. Med. Imaging*, vol. 39, no. 12, pp. 3868–3878, 2020. doi: [10.1109/TMI.2020.3006437](https://doi.org/10.1109/TMI.2020.3006437)

**15.** G. Carneiro, L. Zorron Cheng Tao Pu, R. Singh, and A. Burt, "Deep learning uncertainty and confidence calibration for the five-class polyp classification from colonoscopy," *Med. Image Anal.*, vol. 62, Art. no. 101653, 2020. doi: [10.1016/j.media.2020.101653](https://doi.org/10.1016/j.media.2020.101653)

**16.** M. Abdar, F. Pourpanah, S. Hussain, D. Rezazadegan, L. Liu, M. Ghavamzadeh, et al., "A review of uncertainty quantification in deep learning: Techniques, applications and challenges," *Inf. Fusion*, vol. 76, pp. 243–297, 2021. doi: [10.1016/j.inffus.2021.05.008](https://doi.org/10.1016/j.inffus.2021.05.008)

**17.** A. N. Angelopoulos and S. Bates, "Conformal prediction: A gentle introduction," *Found. Trends Mach. Learn.*, vol. 16, no. 4, pp. 494–591, 2023. doi: [10.1561/2200000101](https://doi.org/10.1561/2200000101)

| No. | Year | Author(s) | Focus | Technique(s) | Key Finding | Gap |
|---|---|---|---|---|---|---|
| 14 | 2020 | Mehrtash et al. | Calibration + uncertainty for medical segmentation | Temp scaling, MC dropout / ensembles | Calibrated confidence in segmentation | Segmentation focus, not lumbar MRI |
| 15 | 2020 | Carneiro et al. | Uncertainty + calibration, 5-class polyp classification | MC dropout / ensembles + calibration | Calibrated confidence in endoscopy | Colonoscopy, not spine |
| 16 | 2021 | Abdar et al. | Review of uncertainty quantification in DL | Survey | Taxonomy of UQ methods and challenges | General, not spine-specific |
| 17 | 2023 | Angelopoulos & Bates | Conformal prediction foundations | Conformal prediction | Distribution-free coverage guarantees | Not applied to lumbar MRI |

---

## Cluster 4 — Lightweight Architectures & Explainability (Papers 18–21)

**18.** B. H. M. van der Velden, H. J. Kuijf, K. G. A. Gilhuijs, and M. A. Viergever, "Explainable artificial intelligence (XAI) in deep learning-based medical image analysis," *Med. Image Anal.*, vol. 79, Art. no. 102470, 2022. doi: [10.1016/j.media.2022.102470](https://doi.org/10.1016/j.media.2022.102470)

**19.** A. Singh, S. Sengupta, and V. Lakshminarayanan, "Explainable deep learning models in medical image analysis," *J. Imaging*, vol. 6, no. 6, Art. no. 52, 2020. doi: [10.3390/jimaging6060052](https://doi.org/10.3390/jimaging6060052)

**20.** J. Kang and J. Gwak, "Ensemble learning of lightweight deep learning models using knowledge distillation for image classification," *Mathematics*, vol. 8, no. 10, Art. no. 1652, 2020. doi: [10.3390/math8101652](https://doi.org/10.3390/math8101652)

**21.** A. Singh, M. P. Singh, and A. K. Singh, "Adversarially enhanced learning (AEL): Robust lightweight deep learning approach for radiology image classification against adversarial attacks," *Image Vis. Comput.*, vol. 154, Art. no. 105405, 2025. doi: [10.1016/j.imavis.2024.105405](https://doi.org/10.1016/j.imavis.2024.105405)

| No. | Year | Author(s) | Focus | Technique(s) | Key Finding | Gap |
|---|---|---|---|---|---|---|
| 18 | 2022 | van der Velden et al. | XAI in DL medical image analysis | Survey (CAMs, etc.) | XAI methods categorized; clinical validation needed | Not integrated with lightweight/uncertainty |
| 19 | 2020 | Singh et al. | Explainable DL in medical images | CAM-based visualization | Explainability improves interpretation | No quantitative uncertainty |
| 20 | 2020 | Kang & Gwak | Lightweight model ensembles | Knowledge distillation | Accurate, efficient ensembles | Not spine, no calibration/XAI |
| 21 | 2025 | Singh et al. | Robust lightweight radiology classification | Lightweight CNN + adversarial learning | Efficient classification robust to attacks | Robustness only, no trust metrics |

---

## Ready-to-paste updates for the deck

- **Lit1 slide** (Papers 1–7): rows 1–7 from the Cluster 1 table (add row 8 → move to Lit2).
- **Lit2 slide** (Papers 8–14): row 8 (Cluster 1) + rows 9–13 (Cluster 2) + row 14 (Cluster 3).
- **Lit3 slide** (Papers 15–21): rows 15–17 (Cluster 3) + rows 18–21 (Cluster 4).
- **References slide**: replace the 10 hard-coded references with the 21 citations above (methodology refs [2]–[10] from your old deck are still valid method citations — see `references_verified.md` notes in this file's history for the 2 that needed fixing: Eigen-CAM 2021→2020, EfficientViT author/pages).
