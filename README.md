# Preeclampsia Risk Prediction

Final Year Project — Master's Degree in Computer Science, Badji Mokhtar Annaba University

---

## Overview

This project aims to predict the risk of preeclampsia in pregnant women using clinical data and three machine learning approaches: Gaussian Mixture Model (GMM), Random Forest, and Support Vector Machine (SVM). Data is stored and retrieved from a MySQL database.

Preeclampsia is a serious pregnancy complication characterized by high blood pressure and proteinuria appearing after the 20th week of pregnancy. Early detection is essential to reduce maternal and fetal risks.

---

## Models

| Model | Type | File |
|---|---|---|
| Gaussian Mixture Model (GMM) | Unsupervised | `entrainement gmm/gmm sql.py` |
| Random Forest | Supervised | `entrainement random forest/random_f.py` |
| Support Vector Machine (SVM) | Supervised | `svm/svm_mysql.py` |
| Comparative Interface | Platform | `interface/platforme deux methode.py` |

---

## Features

| Feature | Description |
|---|---|
| `age` | Patient age |
| `sbp` | Systolic blood pressure |
| `dbp` | Diastolic blood pressure |
| `BMI` | Body mass index |
| `Proteinuria` | Presence of protein in urine |
| `chronic_hypertension` | Chronic hypertension |
| `first_pregnancy` | First pregnancy |
| `after_20_weeks` | Symptoms after 20 weeks |
| `multiple` | Multiple pregnancy |
| `diabetes` | Diabetes |
| `PE` | Target — preeclampsia (0 = No, 1 = Yes) |

---

## Results

### Random Forest
- Training accuracy: **~96%**
- Test accuracy: **~97%**
- Most important feature: **Proteinuria** (0.25), followed by **SBP** (0.22)

### SVM
- Most important feature: **DBP** (0.26), followed by **SBP** (0.24)

### GMM
- Overall accuracy: **~92%**
- Unsupervised clustering with 2 components

---

## Project Structure

```
preeclampsia-prediction/
│
├── database.xlsx                             # Main dataset
│
├── entrainement gmm/
│   ├── gmm sql.py                            # GMM training script
│   ├── gmm_model.pkl                         # Saved GMM model
│   ├── scaler.pkl                            # Saved scaler
│   ├── pregnancy_cases_with_GMM.xlsx         # GMM results
│   └── Figure_1.png                          # GMM confusion matrix
│
├── entrainement random forest/
│   ├── random_f.py                           # Random Forest training script
│   ├── random_forest_preeclampsia_model.pkl  # Saved RF model
│   ├── preeclampsia_prediction_results.xlsx  # RF results
│   ├── feature_importance.png                # Feature importance chart
│   ├── confusion_train_RF.png                # Training confusion matrix
│   └── confusion_test_RF.png                 # Test confusion matrix
│
├── svm/
│   ├── svm_mysql.py                          # SVM training script
│   ├── svm_pe_model.pkl                      # Saved SVM model
│   ├── permutation_importance_*.png          # Permutation importance chart
│   ├── confusion_matrix_training_SVM.png     # Training confusion matrix
│   ├── confusion_matrix_test_SVM.png         # Test confusion matrix
│   ├── predictions_training_*.xlsx           # Training predictions
│   ├── predictions_test_*.xlsx               # Test predictions
│   └── feature_importance_*.xlsx             # Feature importance data
│
└── interface/
    └── platforme deux methode.py             # Comparative model interface
```

---

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib sqlalchemy pymysql mysql-connector-python openpyxl
```

MySQL database required:
- Host: `localhost`
- Database: `PreeclampsiaRiskDB`
- Table: `pregnancy_cases`

---

## Usage

```bash
# Train GMM model
python "entrainement gmm/gmm sql.py"

# Train Random Forest model
python "entrainement random forest/random_f.py"

# Train SVM model
python svm/svm_mysql.py

# Launch comparative interface
python "interface/platforme deux methode.py"
```

---

## Authors

Project developed as part of the Master's Degree in Computer Science — Badji Mokhtar Annaba University

- **Serine Bougheloum** — [github.com/serinebgh](https://github.com/serinebgh)
- **Ouissal Boumendjel** — [github.com/Ouissal-Boumendjel](https://github.com/Ouissal-Boumendjel)
