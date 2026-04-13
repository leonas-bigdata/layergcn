# LayerGCN for Recommendation System

This repository is a fork of the original **ImRec: Implicit feedback based Recommendation framework**.

In this fork, our team will **only keep and use the LayerGCN model**, and ignoring other models in the original framework.

---

## Overview

The goal of this repository is to conduct experiment on **LayerGCN (Layer-refined Graph Convolutional Network)** for implicit feedback-based recommendation with a different approach from the original **RecBold** framework.

---

## Key Features & Changes

- **Focused Model Support**
  - Only supports **LayerGCN** (other models from the original repo are not used in this fork).

- **Different Approach Instead of Timestamp Splitting**
  - We modify the code to suitable for our trainable benchmark datasets format.

- **Reproducibility**
  - Fixed seed setup
  - Deterministic training pipeline (as much as possible under PyTorch/CUDA)

## Model Supported

| Model     | Paper |
|----------|-------|
| LayerGCN | Layer-refined Graph Convolutional Networks for Recommendation (ICDE 2023) |

## Dataset

The model is evaluated on standard implicit feedback datasets such as:

- Gowalla
- Yelp2018
- AmazonBooks
- Movielen

---

## Running Experiment

Train LayerGCN on a dataset

```bash
python main.py -m LayerGCN -d gowalla
```

## Original Repository

This fork is based on the ImRec framework:
- https://github.com/enoche/MMRec


## Citation

If you use this code, please cite the original LayerGCN paper:

```bibtex
@inproceedings{zhou2023layer,
  title={Layer-refined graph convolutional networks for recommendation},
  author={Zhou, Xin and Lin, Donghui and Liu, Yong and Miao, Chunyan},
  booktitle={2023 IEEE 39th International Conference on Data Engineering (ICDE)},
  pages={1247--1259},
  year={2023},
  organization={IEEE}
}
```

---

## License

Same as upstream repository unless otherwise specified.



