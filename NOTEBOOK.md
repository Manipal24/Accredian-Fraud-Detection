# Jupyter Notebook - Accredian Fraud Detection Model

## Live Interactive Notebook

The complete Jupyter notebook for this project is available in Google Colab for interactive exploration and execution:

### [Open Notebook in Google Colab](https://colab.research.google.com/drive/13iVzV8QW75jdm_UHUZtXV_4V98pcbqxh)

## Notebook Contents

The notebook contains:

1. **Project Overview**
   - Objective: Build a fraud detection model for mobile-money transactions
   - Business Context and importance of fraud prevention

2. **Data Exploration & Cleaning**
   - Data loading and initial exploration
   - Missing values analysis and handling
   - Outlier detection and removal using IQR method
   - Multicollinearity analysis using VIF

3. **Feature Engineering**
   - Feature selection methodologies
   - Correlation-based feature selection
   - Recursive Feature Elimination (RFE)

4. **Model Training & Evaluation**
   - Multiple model implementations:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting
   - Performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
   - Model comparison and best model selection

5. **Key Fraud Predictors**
   - Feature importance analysis
   - SHAP values for model interpretability
   - Domain validation of identified predictors

6. **Business Insights & Recommendations**
   - Prevention infrastructure recommendations
   - Real-time monitoring strategies
   - Success measurement KPIs

## How to Use

1. Click the link above to open the notebook in Google Colab
2. Run all cells sequentially or individually
3. Modify parameters and experiment with different approaches
4. Download the notebook as .ipynb for local use

## Requirements

- Python 3.x
- pandas, numpy, scikit-learn, matplotlib, seaborn
- Google Colab (for interactive execution) or Jupyter Notebook (for local execution)

## Note

The notebook uses a simulated financial dataset (PaySim) with 6.36M transactions for demonstration purposes. All analysis and conclusions are based on this simulated data.
