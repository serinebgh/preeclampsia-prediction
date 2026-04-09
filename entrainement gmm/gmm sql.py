import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sqlalchemy import create_engine

# --- MySQL connection info ---
mysql_host = 'localhost'
mysql_user = 'root'
mysql_password = '1234'
mysql_db = 'PreeclampsiaRiskDB'
mysql_table = 'pregnancy_cases'

# --- Connection and data loading from MySQL ---
connection_str = f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
engine = create_engine(connection_str)

query = f"SELECT * FROM {mysql_table}"
data = pd.read_sql(query, engine)

# --- Explanatory variables (quantitative + qualitative) ---
features = ['age', 'sbp', 'dbp', 'multiple', 'diabetes', 'BMI',
            'first_pregnancy', 'after_20_weeks', 'chronic_hypertension',
            'Proteinuria']

X = data[features]
y = data['PE']

# --- Normalization of quantitative variables ---
continuous_vars = ['age', 'sbp', 'dbp', 'BMI']
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[continuous_vars] = scaler.fit_transform(X[continuous_vars])

# --- GMM model training ---
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

# --- Saving the trained GMM model ---
model_path = "gmm_model.pkl"
joblib.dump(gmm, model_path)
print(f"GMM model saved: {model_path}")

# --- Saving the scaler ---
scaler_path = "scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved: {scaler_path}")

# --- Predictions ---
y_pred = gmm.predict(X_scaled)
probs = gmm.predict_proba(X_scaled)

# --- Label adjustment ---
if accuracy_score(y, y_pred) < 0.5:
    y_pred = 1 - y_pred
    probs = probs[:, ::-1]

# --- Adding columns to DataFrame ---
data['Cluster_0_Prob'] = probs[:, 0]
data['Cluster_1_Prob'] = probs[:, 1]
data['Cluster_Pred'] = y_pred

# --- Percentage of each cluster ---
cluster_counts = data['Cluster_Pred'].value_counts(normalize=True) * 100
cluster_percent_df = cluster_counts.reset_index()
cluster_percent_df.columns = ['Cluster', 'Percentage']
cluster_percent_df['Percentage'] = cluster_percent_df['Percentage'].round(2)
print("\nCluster percentages:")
print(cluster_percent_df)

# --- Saving to Excel file ---
output_file = "pregnancy_cases_with_GMM.xlsx"
data.to_excel(output_file, index=False)
print(f"File saved: {output_file}")

# --- Evaluation ---
print("\nGMM evaluation on complete dataset:")
print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
print("\nConfusion matrix:")
print(confusion_matrix(y, y_pred))
print("\nClassification report:")
print(classification_report(y, y_pred))

# --- Confusion matrix visualization ---
cm_df = pd.DataFrame(confusion_matrix(y, y_pred),
                     index=['Actual: PE=0', 'Actual: PE=1'],
                     columns=['Predicted: PE=0', 'Predicted: PE=1'])
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('GMM Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig("gmm_confusion_matrix.png")
plt.show()