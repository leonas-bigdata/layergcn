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

## 📦 Important Task 3 Experiments & Visualize folders

- `exp` folder store our experiments notebooks file

- `plot` folder store our visualization and extracted copied logs from the notebooks files

---

## 🛠️ Key Modifications in This Fork

To make the framework simpler and more suitable for experimentation, we made the following changes:

### 1. Model Cleanup
- Removed all `<model_name>.py` files except `layergcn.py`

### 2. 0.0 values for validation  (`quick_start.py`)
- Since we have configured the the code structure that compatible for the training and testing data only. All of the values from validation output will be 0s

- Example : 

```
Valid: recall@20: 0.0000    ndcg@20: 0.0000    map@20: 0.0000    precision@20: 00000
Test: recall@20: 0.1541    ndcg@20: 0.1231    map@20: 0.0591    precision@20: 0.0454 
```

### 3. Modify logic in `dataset.py` file

- The original strictly require a single dataset file named `(dataset_name).inter`. So it will crash if it was not found. Therefore, we have added the method `_load_user_list_format`, which parse lines formatted as user_id item_id1 item_id2, ... , and flattens them into a proper DataFrame of (user_id, item_id) pairs. It set a flag `self.is_pre_split` = True when using this route

- Update the `split` by ignoring the chronological splitting logic to use the pre-split text files. The 

    - Original: Handled splitting purely by computing timestamps and slicing the single dataframe chronologically to train, valid, test

    - Modified: We added a blockage at the begining of the `split` function. We check if the loaded data is `train.txt`, then we will maps the string IDs to integers Ids across both the train and test dataframes. It defines the `train_ds` and `test_ds` using `train.txt` and `test.txt`

- Ignore K-core collaborative filtering is a must because our already-processed have been applied k-core filtering already. Applying it again is not necessary and could potentially reduce the quality of the dataset.

### 4. Modify logic in `layergcn.py` file

- update the `embeddings_layers`.

    - Original : The `embeddings_layers = []` which is empty. This means the intial ego embedding at layer 0 were never added to the final stack before summiing.

    - Modified : We initialized the list with the ego embeddings `embeddings_layers = [ego_embeddings]` . Now when the model computes `ui_all_embeddings = torch.sum(...)`, it include the initial features along with the propagated features from the graph layers

- Fix `get_norm_adj_mat` for scipy compatibility. The original used the `A._update(data_dict)` which is a deprecated implementation, we change this by modifying it to `dict.update(A, data_dict)`

### 5. Modify logic in `trainer.py` file

- Add system resource monitoring : we import `psutil` lib to log out the RAM and VRAM metrics

- Early stopping logic change to use Test Data

    - Original code evaluated the `valid_score` against `best_valid_score` to determine if the model had stopped improving

    - We modified this by commented out the validation early stopping check, and extract the `test_score = test_result[self.valid_metric]` and pass it to the `early_stopping` function

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

We sincerely thank the authors of the original LayerGCN paper and repository for making their work publicly available.

Please consider citing the original paper when using this codebase.