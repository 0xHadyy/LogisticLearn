# Logistic Regression from Scratch (Math + Numpy) 

[![PDF Notes](https://img.shields.io/badge/PDF-Theory-blue)](docs%20and%20theory/Logistic%20Regression.pdf)
[![Notebook Implementation](https://img.shields.io/badge/Jupyter-Notebook-orange)](notebooks/Logistic_Regression_Scratch.ipynb)
[![YouTube Tutorial](https://img.shields.io/badge/YouTube-Tutorial-red)](https://youtu.be/fQFKdAuTSZ8)

This repository is my full implementation of **Binary Logistic Regression** using only **NumPy** :

<p align="center">
<img src="https://s12.gifyu.com/images/b3z3E.gif" width= "600" >
</p>



- Full implementation of logistic regression 
- **L1(Lasso)** & **L2(Ridge)** Regularization 
- **cross-validation** and **hyperparameter tuning** From Scratch 
- Step-by-Step **Jupyter Notebook** with explanations
- **PDF documentation** of Logistic Regression math and theory
- Benchmark against **scikit-learn**





## 📋Table of Contents
1. [Topics Covered](#topics-covered)
2. [Implementation Details](#implementation-details)
3. [Results](#results)
4. [How to Run](#how-to-run)
5. [Video Explanation](#video-explanation)
6. [References](#references)

## 📚Topics Covered
- **Core Model**
  - Logistic (Sigmoid) Function
  - Log-Likelihood and Cross-Entropy Loss
  - Batch, Stochastic, and Mini-Batch Gradient Descent
  - Newton-Raphson (Iteratively Reweighted Least Squares)
  - Regularization: L1 (Lasso), L2 (Ridge)
- **Model Evaluation**
  - Accuracy, Precision, Recall, F1-Score
  - ROC Curve & AUC
  - Decision Boundary Visualization
- **Cross Validation & Hyperparameter Tuning**
  - K-Fold & Stratified K-Fold Cross Validation
  - Grid Search for Learning Rate & Regularization Strength
- **Benchmarks**
  - Heart Disease Prediction
  - Breast Cancer Prediction
  - Benchmark vs `scikit-learn` LogisticRegression

## ⚙️Implementation Details
- Fully object-oriented design (class-based)
- Implements cloning, model saving, and detailed logging
- Custom `GridSearchCV` and `CrossValidation` implementations
- Proximal operator for L1 regularization
- Supports multiple optimizers and convergence criteria

## 📊Results 

- Comparison between custom implementation and scikit-learn
- L1 vs L2 Regularization Coefficient Analysis
- ROC Curve and Decision Boundary Visualizations
- Performance Metrics across Folds
- Training Time Comparison
- Results from the Heart Disease mini study(add later)


## How to Run
```bash
# Clone the repository
git clone https://github.com/0xHadyy/Logistic-Regression-From-Scratch.git
cd Logistic-Regression-From-Scratch

# Install dependencies
pip install -r requirements.txt

# Run the main notebook
jupyter notebook Logistic_Regression.ipynb

```

## 🎥Video Eplanation 
I also created a video walkthrough explaining :
- The math and intution behind Logistic Regression 
- How to implement it from scratch in Numpy
- Regularization, cross-validation and grid search

## 📚References

🔗 My Machine Leanring Notes GitHub Repo: [Isl-python](https://github.com/0xHadyy/Linear-Regression-From-Scratch)

📘 An Introduction to Statistical Learning — James, Witten, Hastie, Tibshirani

📗 The Elements of Statistical Learning — Hastie, Tibshirani, Friedman
