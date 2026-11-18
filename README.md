# Volleyball Group Activity Recognition

Implementation of the hierarchical temporal baselines (B1 to B8) for group activity recognition on the Volleyball Dataset (*Ibrahim et al., CVPR 2016*).

## Overview

The pipeline classifies 8 group activities (`l-pass`, `r-pass`, `l-spike`, `r_spike`, `l_set`, `r_set`, `l_winpoint`, `r_winpoint`) and 9 individual player actions (`waiting`, `setting`, `digging`, `falling`, `spiking`, `blocking`, `jumping`, `moving`, `standing`) across 9-frame video clips.

## Results Summary (Test Split - 16 Videos)

| Baseline | Model | Description | Test Accuracy |
| :--- | :--- | :--- | :---: |
| **B1** | ResNet50 (Scene) | Full-frame scene classifier (9-frame consensus) | 58.86% |
| **B2** | ResNet50 (Player Crops) | Individual player action classifier (9 classes) | 68.69% |
| **B3** | MLP on Max-Pooled Players | Max pooling over 12 players' B2 features | 67.99% |
| **B4** | Person LSTM | Temporal LSTM per player followed by Max Pooling | 74.87% |
| **B5** | Group LSTM | Max Pooling across players first, then Group LSTM | 69.41% |
| **B6** | Scene GRU | Temporal GRU over B1 scene features | 61.26% |
| **B7** | Two-Stream Fusion (1-Group) | Concatenation of global player pool (512-D) + scene (512-D) | 79.58% |
| **B8** | **Hierarchical Team GRU (2-Group)** | **Separate Left Team (512-D) + Right Team (512-D) + Scene (512-D)** | **88.03%** |

## Repository Structure

- `code.ipynb`: Main notebook containing data preprocessing, EDA, feature extraction, and training/evaluation for baselines B1-B8.
- `models/`: Saved PyTorch weights (`.pth`) for the best checkpoint of each baseline.
