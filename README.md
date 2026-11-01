# Wrapper Feature Selection with a Nearest-Neighbor Classifier

![Python](https://img.shields.io/badge/python-NumPy%20%7C%20pandas-blue)
![Type](https://img.shields.io/badge/CS%20205%20Artificial%20Intelligence-UC%20Riverside-lightgrey)

This project implements a nearest-neighbor classifier from scratch and wraps it in two greedy feature-search strategies, **forward selection** and **backward elimination**. The goal is to find the feature subset that maximizes classification accuracy.

*Project 2 for CS 205 (Artificial Intelligence), UC Riverside. The write-up is in [`Project2_Report_Shahriar (1).pdf`](Project2_Report_Shahriar%20(1).pdf).*

## Method

- **Classifier:** a k-nearest-neighbor classifier (k = 1 by default) with Euclidean distance on min-max-normalized features, evaluated by 10-fold cross-validation.
- **Forward selection:** start from the empty set; at each step, add the feature that gives the highest accuracy.
- **Backward elimination:** start from all features; at each step, remove the feature whose removal gives the highest accuracy.
- **Datasets:** three synthetic datasets of increasing size (small, large and extra-large), plus the real-world Pima Indians Diabetes dataset (`diabetes.csv`).

## Files

| File | Content |
|---|---|
| `knn-feature-selection.ipynb` | Implementation, search traces and accuracy plots |
| `CS170_small_Data__18.txt`, `CS170_large_Data__31.txt`, `CS170_XXXlarge_Data__12.txt` | Synthetic datasets (first column = class label) |
| `diabetes.csv` | Real-world dataset |

## Run

```bash
pip install numpy pandas matplotlib seaborn tqdm
jupyter notebook knn-feature-selection.ipynb
```
