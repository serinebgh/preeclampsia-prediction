import pandas as pd
import numpy as np
import mysql.connector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime

# Database connection configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '1234',
    'database': 'PreeclampsiaRiskDB'
}

def connect_to_database():
    """Establish connection to MySQL database"""
    try:
        connection = mysql.connector.connect(**db_config)
        print("Successfully connected to MySQL database")
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

def find_data_table():
    """Find the table containing preeclampsia data"""
    connection = connect_to_database()
    if connection is None:
        return None
    
    try:
        cursor = connection.cursor()
        
        # List all tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        print("Available tables in the database:")
        for i, table in enumerate(tables):
            print(f"{i+1}. {table[0]}")
        
        if not tables:
            print("No tables found in the database.")
            return None
        
        # Look for a table with the necessary columns
        required_columns = ['age', 'sbp', 'dbp', 'PE']  # Minimum required columns
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"DESCRIBE {table_name}")
            columns = [col[0].lower() for col in cursor.fetchall()]
            
            # Check if required columns are present
            if all(col.lower() in columns for col in required_columns):
                print(f"\nTable '{table_name}' selected (contains required columns)")
                return table_name
        
        # If no automatic table is found, ask the user
        print("\nNo table with required columns found automatically.")
        print("Please check your table name or use the first available table.")
        return tables[0][0] if tables else None
        
    except Exception as e:
        print(f"Error while searching for table: {e}")
        return None
    finally:
        cursor.close()
        connection.close()

def get_table_columns(table_name):
    """Get the columns of a table"""
    connection = connect_to_database()
    if connection is None:
        return []
    
    try:
        cursor = connection.cursor()
        cursor.execute(f"DESCRIBE {table_name}")
        columns = [col[0] for col in cursor.fetchall()]
        return columns
    except Exception as e:
        print(f"Error while retrieving columns: {e}")
        return []
    finally:
        cursor.close()
        connection.close()

def load_data_from_mysql():
    """Load data from MySQL database"""
    # Find the data table
    table_name = find_data_table()
    if table_name is None:
        return None
    
    # Get the table columns
    columns = get_table_columns(table_name)
    print(f"\nAvailable columns in '{table_name}': {columns}")
    
    connection = connect_to_database()
    if connection is None:
        return None
    
    try:
        # Build the query with available columns
        # Desired columns (adapt according to your data)
        desired_columns = ['age', 'sbp', 'dbp', 'multiple', 'diabetes', 'BMI', 
                          'first_pregnancy', 'after_20_weeks', 'chronic_hypertension', 
                          'Proteinuria', 'PE']
        
        # Use only the columns that exist
        available_columns = [col for col in desired_columns if col in columns]
        
        if 'PE' not in available_columns:
            print("Error: The target column 'PE' was not found in the table.")
            return None
        
        query = f"SELECT {', '.join(available_columns)} FROM {table_name}"
        print(f"\nSQL Query: {query}")
        
        df = pd.read_sql(query, connection)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Loaded columns: {list(df.columns)}")
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    finally:
        connection.close()

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    print("\nData information:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for missing values
    print("\nMissing values per column:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    # Handle missing values if any
    if missing_values.sum() > 0:
        print("Removing rows with missing values...")
        df = df.dropna()
        print(f"New dimensions after cleaning: {df.shape}")
    
    # Check that the PE column exists
    if 'PE' not in df.columns:
        print("Error: The 'PE' column (target variable) is not present.")
        return None, None
    
    # Separate features and target
    X = df.drop('PE', axis=1)
    y = df['PE']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target distribution:")
    print(y.value_counts())
    
    return X, y

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    # Initialize Random Forest with good parameters
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced'
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    print("Random Forest model trained successfully")
    
    return rf_model

def evaluate_model(model, X_test, y_test, dataset_name="Test"):
    """Evaluate model performance"""
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (PE)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n{dataset_name} Set Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, dataset_name, save_path=""):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No PE', 'PE'], 
                yticklabels=['No PE', 'PE'])
    plt.title(f'Confusion Matrix - {dataset_name} Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(f"{save_path}/confusion_matrix_{dataset_name.lower()}.png", 
                   dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"confusion_matrix_{dataset_name.lower()}.png", 
                   dpi=300, bbox_inches='tight')
    plt.show()

def get_feature_importance(model, feature_names):
    """Get and display feature importance"""
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (Most Influential Factors):")
    print(feature_importance_df)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x='importance', y='feature')
    plt.title('Feature Importance - Random Forest Model')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance_df

def save_results_to_excel(X_train, X_test, y_train, y_test, 
                         train_results, test_results, feature_importance_df):
    """Save all results to Excel file"""
    
    # Create Excel writer
    with pd.ExcelWriter('preeclampsia_prediction_results.xlsx', engine='openpyxl') as writer:
        
        # Training data with predictions and probabilities
        train_data = X_train.copy()
        train_data['Actual_PE'] = y_train.values
        train_data['Predicted_PE'] = train_results['predictions']
        train_data['PE_Probability'] = train_results['probabilities']
        train_data.to_excel(writer, sheet_name='Training_Data', index=False)
        
        # Test data with predictions and probabilities
        test_data = X_test.copy()
        test_data['Actual_PE'] = y_test.values
        test_data['Predicted_PE'] = test_results['predictions']
        test_data['PE_Probability'] = test_results['probabilities']
        test_data.to_excel(writer, sheet_name='Test_Data', index=False)
        
        # Model performance metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Training_Set': [train_results['accuracy'], train_results['precision'], 
                           train_results['recall'], train_results['f1_score']],
            'Test_Set': [test_results['accuracy'], test_results['precision'], 
                        test_results['recall'], test_results['f1_score']]
        })
        metrics_df.to_excel(writer, sheet_name='Model_Performance', index=False)
        
        # Feature importance
        feature_importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
        
        # Confusion matrices
        train_cm_df = pd.DataFrame(train_results['confusion_matrix'], 
                                 columns=['Predicted_No_PE', 'Predicted_PE'],
                                 index=['Actual_No_PE', 'Actual_PE'])
        train_cm_df.to_excel(writer, sheet_name='Train_Confusion_Matrix')
        
        test_cm_df = pd.DataFrame(test_results['confusion_matrix'], 
                                columns=['Predicted_No_PE', 'Predicted_PE'],
                                index=['Actual_No_PE', 'Actual_PE'])
        test_cm_df.to_excel(writer, sheet_name='Test_Confusion_Matrix')
    
    print("Results saved to preeclampsia_prediction_results.xlsx")

def save_model(model, filename="random_forest_preeclampsia_model.pkl"):
    """Save trained model to pickle file"""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")

def main():
    """Main execution function"""
    print("=== Preeclampsia Risk Prediction using Random Forest ===\n")
    
    # Load data from MySQL
    print("1. Loading data from MySQL...")
    df = load_data_from_mysql()
    
    if df is None:
        print("Failed to load data. Please check your database connection and table.")
        return
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    X, y = preprocess_data(df)
    
    if X is None or y is None:
        print("Failed to preprocess data.")
        return
    
    # Split data (71% training, 29% testing)
    print("\n3. Splitting data (71% training, 29% testing)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Train Random Forest model
    print("\n4. Training Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)
    
    # Evaluate on training set
    print("\n5. Evaluating model performance...")
    train_results = evaluate_model(rf_model, X_train, y_train, "Training")
    
    # Evaluate on test set
    test_results = evaluate_model(rf_model, X_test, y_test, "Test")
    
    # Plot confusion matrices
    print("\n6. Generating confusion matrices...")
    plot_confusion_matrix(train_results['confusion_matrix'], "Training")
    plot_confusion_matrix(test_results['confusion_matrix'], "Test")
    
    # Get feature importance
    print("\n7. Analyzing feature importance...")
    feature_importance_df = get_feature_importance(rf_model, X.columns)
    
    # Save results to Excel
    print("\n8. Saving results to Excel...")
    save_results_to_excel(X_train, X_test, y_train, y_test, 
                         train_results, test_results, feature_importance_df)
    
    # Save trained model
    print("\n9. Saving trained model...")
    save_model(rf_model)
    
    print("\n=== Analysis Complete ===")
    print("Files generated:")
    print("- preeclampsia_prediction_results.xlsx (Complete results)")
    print("- random_forest_preeclampsia_model.pkl (Trained model)")
    print("- confusion_matrix_training.png")
    print("- confusion_matrix_test.png")
    print("- feature_importance.png")

if __name__ == "__main__":
    main()