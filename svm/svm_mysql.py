import mysql.connector
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, roc_curve, auc,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sqlalchemy import create_engine
import os
from datetime import datetime

# Configuration
pd.set_option('display.max_columns', None)
np.set_printoptions(precision=3)

# Create output directory
OUTPUT_DIR = "C:/random amel/svm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_feature_importance(model, feature_names):
    """Get and display feature importance for SVM"""
    if hasattr(model, 'coef_') and model.coef_ is not None:
        # For linear SVM, use coefficients
        if len(model.coef_.shape) > 1:
            importance_raw = np.abs(model.coef_[0])
        else:
            importance_raw = np.abs(model.coef_)
        
        # Normalize to sum to 1.0 (like Random Forest)
        importance_normalized = importance_raw / importance_raw.sum()
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_normalized,
            'raw_coefficient': importance_raw
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance (Most Influential Factors):")
        print("Note: Importance values normalized to sum to 1.0 for better comparison")
        print(feature_importance_df[['feature', 'importance']].round(4))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df, x='importance', y='feature', palette='viridis')
        plt.title('Feature Importance - SVM Model (Normalized Linear Coefficients)')
        plt.xlabel('Normalized Importance Score (sum = 1.0)')
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        importance_filename = os.path.join(OUTPUT_DIR, f'feature_importance_{timestamp}.png')
        plt.savefig(importance_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance_df
    
    else:
        print("\nNote: This SVM model (non-linear kernel) doesn't provide direct feature importance.")
        print("Using permutation importance instead...")
        
        # Alternative: Use permutation importance
        from sklearn.inspection import permutation_importance
        
        # This requires X and y data, so we'll return None and handle it in main
        return None

def load_data_from_mysql():
    """Load data from MySQL using SQLAlchemy (recommended)"""
    try:
        # Using SQLAlchemy (avoids pandas warning)
        engine = create_engine('mysql+pymysql://root:1234@localhost/PreeclampsiaRiskDB')
        
        query = """
        SELECT age, sbp, dbp, multiple, diabetes, BMI, first_pregnancy, 
               after_20_weeks, chronic_hypertension, Proteinuria, 
               PE
        FROM pregnancy_cases
        """
        
        df = pd.read_sql(query, engine)
        engine.dispose()
        
    except Exception as e:
        print(f"SQLAlchemy error, using mysql.connector: {e}")
        # Fallback to mysql.connector
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='1234',
            database='PreeclampsiaRiskDB'
        )
        
        query = """
        SELECT age, sbp, dbp, multiple, diabetes, BMI, first_pregnancy, 
               after_20_weeks, chronic_hypertension, Proteinuria, 
               PE
        FROM pregnancy_cases
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
    
    # Convert boolean columns to int 0/1
    bool_cols = ['multiple', 'diabetes', 'first_pregnancy', 'after_20_weeks',
                 'chronic_hypertension', 'Proteinuria', 'PE']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    return df

def analyze_data(data):
    """Exploratory data analysis"""
    print("\n=== DATA ANALYSIS ===")
    print(f"Dataset shape: {data.shape}")
    print(f"\nTarget variable distribution (PE):")
    print(data['PE'].value_counts())
    print(f"Percentage of positive cases: {data['PE'].mean()*100:.1f}%")
    
    print(f"\nMissing values:")
    print(data.isnull().sum())
    
    print(f"\nDescriptive statistics:")
    print(data.describe())

def split_data(data, test_size=0.30):
    """Split data into train/test sets"""
    features = [
        'age', 'sbp', 'dbp', 'multiple', 'diabetes', 'BMI',
        'first_pregnancy', 'after_20_weeks', 'chronic_hypertension',
        'Proteinuria'
    ]
    
    # Check that all features exist
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        features = [f for f in features if f in data.columns]
    
    X = data[features]
    y = data['PE']  # Keep PE as target variable
    
    # Check if there are enough positive cases for stratification
    if y.sum() < 2:
        print("Warning: Very few positive cases, no stratification")
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

def normalize_data(X_train, X_test):
    """Normalize continuous features"""
    # Create copies to avoid warnings
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()
    
    scaler = StandardScaler()
    cont_features = ['age', 'sbp', 'dbp', 'BMI']
    
    # Check that continuous features exist
    cont_features = [f for f in cont_features if f in X_train.columns]
    
    if cont_features:
        X_train_norm[cont_features] = scaler.fit_transform(X_train[cont_features])
        X_test_norm[cont_features] = scaler.transform(X_test[cont_features])
    
    return X_train_norm, X_test_norm, scaler

def train_model_with_validation(X_train, y_train):
    """Train model with cross-validation"""
    from sklearn.model_selection import cross_val_score
    
    # Test different hyperparameters - Include linear kernel for feature importance
    models = [
        ('SVM_Linear', SVC(kernel='linear', probability=True, random_state=42)),
        ('SVM_RBF_C1', SVC(C=1.0, kernel='rbf', probability=True, random_state=42)),
        ('SVM_RBF_C10', SVC(C=10.0, kernel='rbf', probability=True, random_state=42)),
    ]
    
    best_score = 0
    best_model = None
    best_name = ""
    
    print("\n=== MODEL SELECTION ===")
    for name, model in models:
        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        print(f"{name}: {mean_score:.3f} (+/- {scores.std() * 2:.3f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name
    
    # Train the best model
    best_model.fit(X_train, y_train)
    print(f"\nBest model selected: {best_name}")
    return best_model

def get_permutation_importance(model, X, y, feature_names):
    """Get permutation importance for non-linear models"""
    from sklearn.inspection import permutation_importance
    
    print("\nCalculating permutation importance...")
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    
    # Normalize importance scores to sum to 1.0 (like Random Forest)
    importance_raw = perm_importance.importances_mean
    importance_normalized = importance_raw / importance_raw.sum() if importance_raw.sum() > 0 else importance_raw
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_normalized,
        'raw_importance': importance_raw,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (Permutation-based):")
    print("Note: Importance values normalized to sum to 1.0 for better comparison")
    print(feature_importance_df[['feature', 'importance', 'std']].round(4))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'][::-1])
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'][::-1])
    plt.xlabel('Normalized Permutation Importance (sum = 1.0)')
    plt.title('Feature Importance - SVM Model (Normalized Permutation-based)')
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    importance_filename = os.path.join(OUTPUT_DIR, f'permutation_importance_{timestamp}.png')
    plt.savefig(importance_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance_df

def evaluate_model(model, X, y, title=""):
    """Evaluate the model"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    print(f"\n=== {title.upper()} ===")
    print(f"Accuracy: {accuracy_score(y, y_pred):.3f}")
    print("\nClassification report:")
    print(classification_report(y, y_pred))

    # Confusion matrix only
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No PE', 'PE'],
                yticklabels=['No PE', 'PE'])
    plt.title(f'{title} - Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save confusion matrix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f'confusion_matrix_{title.lower()}_{timestamp}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return y_pred, y_proba

def save_predictions_to_excel(X, y_true, y_pred, y_proba, title="predictions"):
    """Save predictions and probabilities to Excel file with detailed metrics"""
    
    # Create predictions DataFrame
    predictions_df = X.copy()
    predictions_df['Actual_PE'] = y_true
    predictions_df['Predicted_PE'] = y_pred
    predictions_df['Risk_Category'] = ['High Risk' if pred == 1 else 'Low Risk' for pred in y_pred]
    
    if y_proba is not None:
        predictions_df['Probability_High_Risk'] = y_proba
        predictions_df['Probability_Low_Risk'] = 1 - y_proba
    
    # Add prediction accuracy column
    predictions_df['Correct_Prediction'] = (y_true == y_pred)
    
    # Calculate detailed metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Save to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = os.path.join(OUTPUT_DIR, f'predictions_{title}_{timestamp}.xlsx')
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Main predictions sheet
        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Enhanced summary statistics sheet
        summary_df = pd.DataFrame({
            'Metric': [
                'Total Cases', 
                'Correct Predictions', 
                'Accuracy', 
                'Precision',
                'Recall (Sensitivity)',
                'F1-Score',
                'Specificity',
                '',  # Empty row for separation
                'True Positives (TP)', 
                'True Negatives (TN)', 
                'False Positives (FP)', 
                'False Negatives (FN)',
                '',  # Empty row for separation
                'High Risk Cases (Actual)', 
                'High Risk Cases (Predicted)',
                'Low Risk Cases (Actual)',
                'Low Risk Cases (Predicted)'
            ],
            'Value': [
                len(y_true),
                sum(y_true == y_pred),
                f"{accuracy:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                f"{tn/(tn+fp) if (tn+fp) > 0 else 0:.4f}",  # Specificity
                '',  # Empty row
                int(tp),
                int(tn),
                int(fp),
                int(fn),
                '',  # Empty row
                sum(y_true == 1),
                sum(y_pred == 1),
                sum(y_true == 0),
                sum(y_pred == 0)
            ],
            'Description': [
                'Total number of cases evaluated',
                'Number of correctly predicted cases',
                'Overall accuracy (TP+TN)/(TP+TN+FP+FN)',
                'Precision = TP/(TP+FP) - Of predicted positive, how many were correct',
                'Recall = TP/(TP+FN) - Of actual positive, how many were detected',
                'F1-Score = 2*(Precision*Recall)/(Precision+Recall) - Harmonic mean',
                'Specificity = TN/(TN+FP) - Of actual negative, how many were correct',
                '',
                'Correctly predicted high risk cases',
                'Correctly predicted low risk cases', 
                'Incorrectly predicted as high risk',
                'Missed high risk cases',
                '',
                'Actual preeclampsia cases in dataset',
                'Cases predicted as high risk',
                'Actual non-preeclampsia cases in dataset',
                'Cases predicted as low risk'
            ]
        })
        summary_df.to_excel(writer, sheet_name='Detailed_Metrics', index=False)
        
        # Performance interpretation sheet
        interpretation_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity'],
            'Current_Value': [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", 
                            f"{f1:.4f}", f"{tn/(tn+fp) if (tn+fp) > 0 else 0:.4f}"],
            'Interpretation': [
                'Good' if accuracy >= 0.8 else 'Moderate' if accuracy >= 0.7 else 'Needs Improvement',
                'Good' if precision >= 0.8 else 'Moderate' if precision >= 0.7 else 'Needs Improvement',
                'Good' if recall >= 0.8 else 'Moderate' if recall >= 0.7 else 'Needs Improvement',
                'Good' if f1 >= 0.8 else 'Moderate' if f1 >= 0.7 else 'Needs Improvement',
                'Good' if (tn/(tn+fp) if (tn+fp) > 0 else 0) >= 0.8 else 'Moderate' if (tn/(tn+fp) if (tn+fp) > 0 else 0) >= 0.7 else 'Needs Improvement'
            ],
            'Clinical_Importance': [
                'Overall model performance',
                'How reliable are positive predictions - reduces false alarms',
                'How well does model catch actual PE cases - critical for patient safety',
                'Balance between precision and recall',
                'How well does model identify low-risk cases'
            ]
        })
        interpretation_df.to_excel(writer, sheet_name='Performance_Analysis', index=False)
    
    print(f"Enhanced predictions saved to Excel: {excel_filename}")
    print(f"Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return excel_filename

def save_model(model, scaler, features, filename='svm_pe_model.pkl'):
    """Save model with metadata"""
    model_path = os.path.join(OUTPUT_DIR, filename)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'timestamp': pd.Timestamp.now(),
        'model_type': 'SVM',
        'model_class': str(type(model))
    }
    
    # Verify model has predict method before saving
    if not hasattr(model, 'predict'):
        raise ValueError(f"Model object does not have predict method. Type: {type(model)}")
    
    print(f"Saving model of type: {type(model)}")
    print(f"Model has predict method: {hasattr(model, 'predict')}")
    print(f"Model has predict_proba method: {hasattr(model, 'predict_proba')}")
    
    joblib.dump(model_data, model_path)
    print(f"Model saved to {model_path}")
    
    # Verify saved model
    try:
        loaded_test = joblib.load(model_path)
        print(f"Verification - Loaded model type: {type(loaded_test['model'])}")
        print(f"Verification - Has predict method: {hasattr(loaded_test['model'], 'predict')}")
    except Exception as e:
        print(f"Error verifying saved model: {e}")
    
    return model_path

def predict_case(model_path, input_data):
    """Predict a new case"""
    try:
        loaded = joblib.load(model_path)
        
        # Extract model and verify it's correct
        model = loaded['model']
        scaler = loaded['scaler']
        expected_features = loaded.get('features', [])
        
        print(f"Loaded model type: {type(model)}")
        print(f"Model has predict method: {hasattr(model, 'predict')}")
        
        if not hasattr(model, 'predict'):
            raise ValueError(f"Loaded model does not have predict method. Type: {type(model)}")
        
        # Create DataFrame with correct features
        df = pd.DataFrame([input_data])
        
        # Check for missing features
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            for feature in missing_features:
                df[feature] = 0  # Default value
        
        # Reorder columns
        df = df[expected_features]
        
        # Normalize continuous features
        cont_features = ['age', 'sbp', 'dbp', 'BMI']
        cont_features = [f for f in cont_features if f in df.columns]
        if cont_features:
            df[cont_features] = scaler.transform(df[cont_features])

        prediction = model.predict(df)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(df)[0]
        else:
            # Fallback for models without predict_proba
            probability = [1-prediction, prediction]

        return {
            'prediction': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability_high_risk': round(float(probability[1]), 3),
            'probability_low_risk': round(float(probability[0]), 3)
        }
        
    except Exception as e:
        return {'error': f"Error during prediction: {str(e)}"}

def main():
    print("=== SVM TRAINING FOR PREECLAMPSIA PREDICTION ===")

    # Load and analyze data
    data = load_data_from_mysql()
    analyze_data(data)
    
    # Critical verification
    if data['PE'].sum() == 0:
        print("ERROR: No positive preeclampsia cases found!")
        return
    
    if len(data) < 50:
        print("WARNING: Very small dataset, results may be unreliable")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(data)
    print(f"\nTrain size: {len(X_train)}, test size: {len(X_test)}")
    print(f"Positive cases train: {y_train.sum()}, test: {y_test.sum()}")
    
    # Normalize
    X_train_norm, X_test_norm, scaler = normalize_data(X_train, X_test)
    
    # Train model
    model = train_model_with_validation(X_train_norm, y_train)
    
    # VERIFY MODEL BEFORE SAVING
    print(f"\nTrained model type: {type(model)}")
    print(f"Model has predict method: {hasattr(model, 'predict')}")
    print(f"Model has predict_proba method: {hasattr(model, 'predict_proba')}")
    
    # GET FEATURE IMPORTANCE
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    feature_names = X_train.columns.tolist()
    importance_df = get_feature_importance(model, feature_names)
    
    # If linear coefficients are not available, use permutation importance
    if importance_df is None:
        importance_df = get_permutation_importance(model, X_train_norm, y_train, feature_names)
    
    # Save feature importance to Excel
    if importance_df is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        importance_filename = os.path.join(OUTPUT_DIR, f'feature_importance_{timestamp}.xlsx')
        importance_df.to_excel(importance_filename, index=False)
        print(f"Feature importance saved to: {importance_filename}")
    
    # Evaluate
    y_train_pred, y_train_proba = evaluate_model(model, X_train_norm, y_train, title="Training")
    y_test_pred, y_test_proba = evaluate_model(model, X_test_norm, y_test, title="Test")
    
    # Save predictions to Excel with enhanced metrics
    save_predictions_to_excel(X_train, y_train, y_train_pred, y_train_proba, title="training")
    save_predictions_to_excel(X_test, y_test, y_test_pred, y_test_proba, title="test")
    
    # Save model
    model_path = save_model(model, scaler, X_train.columns.tolist())
    
    # Prediction test
    new_case = {
        'age': 35, 'sbp': 140, 'dbp': 90, 'multiple': 0,
        'diabetes': 1, 'BMI': 28.4, 'first_pregnancy': 0,
        'after_20_weeks': 1, 'chronic_hypertension': 1,
        'Proteinuria': 1  
    }
    
    result = predict_case(model_path, new_case)
    print("\n=== NEW CASE PREDICTION ===")
    print(result)

if __name__ == "__main__":
    # Install dependencies if necessary
    try:
        import pymysql
    except ImportError:
        print("To use SQLAlchemy with MySQL, install: pip install pymysql sqlalchemy")
    
    main()