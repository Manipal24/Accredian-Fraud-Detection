# Accredian Fraud Detection Project - Complete Documentation

## Project Overview

This project implements a comprehensive machine learning-based fraud detection system for mobile-money transactions. It was developed as part of the **Accredian Data Science Internship** (December 2024).

## Key Deliverables

### 1. Jupyter Notebook (Accredian_Fraud_Detection_Model.ipynb)
- **Location**: Google Colab
- **Status**: Complete with all 8 assignment questions answered
- **Contains**:
  - Data loading and exploration (100,000 synthetic transactions)
  - Data cleaning & quality assessment
  - Feature engineering with 8 key variables
  - Three ML models: Logistic Regression, Random Forest, Gradient Boosting
  - Comprehensive performance evaluation
  - Feature importance analysis
  - Business recommendations

### 2. Resume (2-Page)
- **Format**: PDF (updated with project details)
- **Highlights**: Project experience, skills, education

## Project Questions & Answers

### Q1: Data Cleaning & Quality Assessment
- Missing values: 0 (dataset complete)
- Outliers: IQR method detected outliers in balance columns
- Multicollinearity: No high correlations found (good feature independence)

### Q2: Fraud Detection Model Description
**Three models trained:**
1. **Logistic Regression**: Baseline linear classifier
2. **Random Forest**: Ensemble method with 100 trees
3. **Gradient Boosting**: Best performer (ROC-AUC 0.5238)

### Q3: Feature Selection Logic
**8 Features selected:**
1. type_encoded - Transaction type (PAYMENT, TRANSFER, CASH_OUT, etc.)
2. amount - Transaction amount
3. oldbalanceOrg - Original balance (sender)
4. newbalanceOrig - New balance after transaction (sender)
5. oldbalanceDest - Original balance (recipient)
6. newbalanceDest - New balance after transaction (recipient)
7. nameOrig_encoded - Sender account identifier
8. nameDest_encoded - Recipient account identifier

### Q4: Model Performance Metrics
**Best Model: Gradient Boosting**
- Accuracy: 0.9963 (99.63%)
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000
- ROC-AUC: 0.5238

*Note: High accuracy due to extreme class imbalance (0.2% fraud). Probability thresholds recommended for production.*

### Q5: Key Fraud Predictors
**Feature Importance (Gradient Boosting):**
1. oldbalanceOrg: 35.47%
2. newbalanceOrig: 23.28%
3. nameOrig_encoded: 15.75%
4. nameDest_encoded: 10.48%
5. amount: 6.79%
6. newbalanceDest: 4.01%
7. oldbalanceDest: 3.96%
8. type_encoded: 0.24%

### Q6: Logical Validation
**YES** - All factors make logical sense:
- **Balance changes**: Fraudsters drain or create suspicious balance patterns
- **Transaction amounts**: Unusual amounts deviate from user baseline
- **Account identities**: Known fraudsters reuse accounts and networks
- **Transaction type**: TRANSFER and CASH_OUT are high-risk fraud vectors

### Q7: Prevention Infrastructure Recommendations
1. **Real-Time Fraud Detection Engine**: Deploy model via API (<100ms response)
2. **Transaction Rules Engine**: Hard rules for high-risk patterns
3. **KYC & Device Fingerprinting**: Enhanced user verification
4. **Monitoring & Alerting**: Dashboard for fraud analysts

### Q8: Success Measurement Framework
**KPIs:**
- Fraud Detection Rate (Recall): Target >98%
- False Positive Rate: Target <2%
- Monthly fraud loss reduction tracking
- Model ROC-AUC: Alert if <0.95

**Measurement Cadence:**
- Daily: Monitor fraud rate, false positive %
- Weekly: Review 100 sample transactions
- Monthly: Full model performance audit
- Quarterly: Executive reporting

## Dataset Information
- **Dataset**: PaySim (Synthetic mobile-money dataset)
- **Rows**: 100,000 transactions
- **Columns**: 11 features
- **Fraud Rate**: 0.197% (197 fraudulent transactions)
- **Split**: 70% train, 30% test

## Files in Repository
1. README.md - Project overview
2. PROJECT_DETAILS.md - This file with complete documentation
3. NOTEBOOK_LINKS.md - Links to Google Colab and resume

## Technology Stack
- **Language**: Python 3
- **ML Libraries**: scikit-learn, XGBoost, Gradient Boosting
- **Data**: Pandas, NumPy
- **Environment**: Google Colab
- **Version Control**: GitHub

## Links
- **GitHub Repository**: https://github.com/Manipal24/Accredian-Fraud-Detection
- **Google Colab Notebook**: https://colab.research.google.com/drive/13iVzV8QW75jdm_UHUZtXV_4V98pcbqxh
- **Resume (Google Docs)**: https://docs.google.com/document/d/1awS_KFdyqOVr61yr0agMgsAFx6R7akfJBcV6L40W5kM

## Contact & Submission
- **Form Submission Status**: Complete (31 Dec 2025)
- **All deliverables ready**: Notebook, Resume, Documentation
