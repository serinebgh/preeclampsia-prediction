# Preeclampsia Risk Prediction

Projet de Fin d'Études — Master Informatique, Université Badji Mokhtar Annaba


## Présentation

Ce projet a pour objectif de prédire le risque de prééclampsie chez les femmes enceintes à partir de données cliniques, en utilisant trois approches de machine learning : GMM, Random Forest et SVM. Les données sont stockées et récupérées depuis une base de données MySQL.

La prééclampsie est une complication grave de la grossesse caractérisée par une hypertension artérielle et une protéinurie apparaissant après la 20ème semaine de grossesse. Une détection précoce est essentielle pour réduire les risques maternels et fœtaux.

---

## Modèles implémentés

| Modèle | Type | Fichier |
|---|---|---|
| Gaussian Mixture Model (GMM) | Non supervisé | `entrainement gmm/gmm sql.py` |
| Random Forest | Supervisé | `entrainement random forest/random_f.py` |
| Support Vector Machine (SVM) | Supervisé | `svm/svm_mysql.py` |
| Interface comparative | Plateforme | `interface/platforme deux methode.py` |

---

## Variables utilisées

| Variable | Description |
|---|---|
| `age` | Âge de la patiente |
| `sbp` | Pression artérielle systolique |
| `dbp` | Pression artérielle diastolique |
| `BMI` | Indice de masse corporelle |
| `Proteinuria` | Présence de protéines dans les urines |
| `chronic_hypertension` | Hypertension chronique |
| `first_pregnancy` | Première grossesse |
| `after_20_weeks` | Symptômes après 20 semaines |
| `multiple` | Grossesse multiple |
| `diabetes` | Diabète |
| `PE` | Cible — prééclampsie (0 = Non, 1 = Oui) |

---

## Résultats

### Random Forest
- Accuracy entraînement : **~96 %**
- Accuracy test : **~97 %**
- Variable la plus importante : **Protéinurie** (0.25), suivie de **SBP** (0.22)

### SVM
- Variable la plus importante : **DBP** (0.26), suivie de **SBP** (0.24)

### GMM
- Accuracy globale : **~92 %**
- Clustering non supervisé en 2 composantes

---

## Structure du projet

```
preeclampsia-prediction/
│
├── database.xlsx                          # Base de données principale
│
├── entrainement gmm/
│   ├── gmm sql.py                         # Entraînement GMM
│   ├── gmm_model.pkl                      # Modèle GMM sauvegardé
│   ├── scaler.pkl                         # Normaliseur sauvegardé
│   ├── pregnancy_cases_with_GMM.xlsx      # Résultats GMM
│   └── Figure_1.png                       # Matrice de confusion GMM
│
├── entrainement random forest/
│   ├── random_f.py                        # Entraînement Random Forest
│   ├── random_forest_preeclampsia_model.pkl
│   ├── preeclampsia_prediction_results.xlsx
│   ├── feature_importance.png
│   ├── confusion_train_RF.png
│   └── confusion_test_RF.png
│
├── svm/
│   ├── svm_mysql.py                       # Entraînement SVM
│   ├── svm_pe_model.pkl
│   ├── permutation_importance_*.png
│   ├── confusion_matrix_training_SVM.png
│   ├── confusion_matrix_test_SVM.png
│   ├── predictions_training_*.xlsx
│   ├── predictions_test_*.xlsx
│   └── feature_importance_*.xlsx
│
└── interface/
    └── platforme deux methode.py          # Interface comparative des modèles
```

---

## Prérequis

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib sqlalchemy pymysql mysql-connector-python openpyxl
```

Base de données MySQL requise :
- Hôte : `localhost`
- Base : `PreeclampsiaRiskDB`
- Table : `pregnancy_cases`

---

## Utilisation

```bash
# Entraînement GMM
python "entrainement gmm/gmm sql.py"

# Entraînement Random Forest
python "entrainement random forest/random_f.py"

# Entraînement SVM
python svm/svm_mysql.py

# Lancer l'interface comparative
python "interface/platforme deux methode.py"
```

---

## Auteurs

Projet réalisé dans le cadre du Master Informatique — Université Badji Mokhtar Annaba

- **Serine** — [github.com/serinebgh](https://github.com/serinebgh)
- **Ouissal Boumendjel** — [github.com/Ouissal-Boumendjel](https://github.com/Ouissal-Boumendjel)
