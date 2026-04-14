# LayerGCN-PyTorch

This repository is our team’s fork of the original **ImRec: Implicit Feedback based Recommendation framework**.

In this fork, **we exclusively retain and experiment with the LayerGCN model**, while removing all other models from the original framework.

> **ICDE 2023** — Xin Zhou, Donghui Lin, Yong Liu, Chunyan Miao.  
> **Layer-refined Graph Convolutional Networks for Recommendation**

- Original ImRec repository: https://github.com/enoche/ImRec

---

## 📌 Overview

**LayerGCN** enhances traditional GCN-based collaborative filtering by refining how embeddings are propagated and aggregated across layers.

This model is particularly effective for **implicit feedback recommendation**, where only user–item interactions are available.

Our fork focuses on:

- Isolating LayerGCN from the large ImRec framework
- Simplifying data loading for adjacency-list datasets
- Ensuring clean, reproducible experiments
- Running and validating experiments on Google Colab

---

## 🛠️ Key Modifications in This Fork

To make the framework simpler and more suitable for experimentation, we made the following changes:

### 1. Model Cleanup
- Removed all `<model_name>.py` files except `layergcn.py`

### 2. Custom Data Loading (`quick_start.py`)
- Added a custom function & update some orignal function to match the dataset structure

---

## 📂 Supported Datasets

This implementation supports four widely used benchmark datasets and one custom dataset prepared by our team:

| Dataset       | Description                                  |
|---------------|----------------------------------------------|
| **Gowalla**   | Location-based social network check-ins      |
| **Yelp2018**  | Business reviews and user ratings            |
| **Amazon-Book** | Book purchase and rating records          |
| **MovieLens1M** | Movie rating dataset                      |
| **GitStar**   | Self-crawled dataset based on GitHub stars  |

---

## 🧪 Experiments

We conducted careful experiments using the PyTorch version of LayerGCN.  
All experimental runs, logs, and metric tracking are stored in the `exp/` directory as Jupyter notebooks.

To ensure transparency and academic integrity, we provide public Google Drive links containing:

- The source code from this repository that we uploaded on Google Drive to mount for experiment running
- Execution logs
- Output results
- Saved checkpoints and metrics

These materials serve as **self-proof** that all experiments were conducted by our team on **Google Colab**, without reusing results from external sources. Accessing each link will lead to a folder with three folder LightGCN, UltraGCN and LayerGCN. The source code for this repository is uploaded inside the LayerGCN folder for all three links

### Environment

- Experiments were executed on **Google Colab**
- GPU-enabled runtime
- Repeated runs to ensure reproducibility and stability

### Experiment Notebooks

| Notebook             | Description                        | Link |
|----------------------|------------------------------------|------|
| `layergcn_exp`       | Initial experimental run           | https://drive.google.com/drive/folders/1Iq_NvTZkTy8MYP-0DPvymDHFO2nDsFK_?usp=drive_link |
| `layergcn_rerun_1`   | Run the second time        | https://drive.google.com/drive/folders/1l93GDNQzRcL9pg4eR6vriTOxrzcUY_G5?usp=drive_link |
| `layergcn_rerun_2`   | Run the third time | https://drive.google.com/drive/folders/12VtE9kucBlikg8x8cVtrnuwBRTTBC68d?usp=drive_link |

These notebooks contain:

- Training configurations
- Evaluation metrics (Recall, NDCG, etc.)
- Result comparisons across runs

---

## 🎯 Purpose of This Fork

This fork was created for:

- Academic study and experimentation across 5 datasets, each dataset is executed three times to get the average score 
- Understanding LayerGCN behavior across datasets
- Ensuring reproducible results
- Extending the original implementation with additional datasets and structured experiment tracking

---

## 🙏 Acknowledgements

We sincerely thank the authors of the original LightGCN paper and repository for making their work publicly available.

Please consider citing the original paper when using this codebase.