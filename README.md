# Logistic Regression from Scratch (Math + Numpy) 

This repository is my full implementaton of **Binary Logistic Regression** using only **NumPy** :

- Full implementation of logistic regression 
- **L1(Lasso)** & **L2(Ridge)** Regularization 
- **cross-validaiton** and **hyperparameter tuning** From Scratch 
- Step-by-Step **Jupyter Notebook** with explanations
- **PDF documentation** of Logistic Regression math and theory
- Benchmark against **sickit-learn**- 

All as part of my journey studying **Machine Learning.**

## 📋 Table of Contents
1. [Topics Covered](#topics-covered)
2. [Implementation Details](#implementation-details)
3. [Results](#results)
4. [How to Run](#how-to-run)
5. [References](#references)

## 📚 Topics Covered
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

## ⚙️ Implementation Details
- Fully object-oriented design (class-based)
- Implements cloning, model saving, and detailed logging
- Custom `GridSearchCV` and `CrossValidation` implementations
- Proximal operator for L1 regularization
- Supports multiple optimizers and convergence criteria

## 📊 Results 

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

- Steps to run the code and benchmark(add later)


## 📚 References

🔗 My Machine Leanring Notes GitHub Repo: [Isl-python](https://github.com/0xHadyy/Linear-Regression-From-Scratch)

📘 An Introduction to Statistical Learning — James, Witten, Hastie, Tibshirani

📗 The Elements of Statistical Learning — Hastie, Tibshirani, Friedman

