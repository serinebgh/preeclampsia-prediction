import tkinter as tk
from tkinter import messagebox
import pickle
import joblib  # Ajout de joblib pour charger le modèle SVM
import numpy as np
import pandas as pd
import mysql.connector

# Loading the Random Forest model
try:
    with open("C:/pfe/entrainement random forest/random_forest_preeclampsia_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    print(f"Random Forest model loaded successfully. Type: {type(rf_model)}")
    # Verify it's a proper model
    if not hasattr(rf_model, 'predict'):
        print("ERROR: Random Forest model doesn't have predict method!")
        exit()
except Exception as e:
    print(f"Error loading Random Forest model: {e}")
    exit()

# Loading the SVM model - CORRECTION PRINCIPALE
try:
    # Le modèle SVM est sauvegardé comme un dictionnaire avec joblib
    svm_data = joblib.load("C:/pfe/svm/svm_pe_model.pkl")
    
    # Extraire le modèle du dictionnaire
    if isinstance(svm_data, dict) and 'model' in svm_data:
        svm_model = svm_data['model']
        svm_scaler = svm_data.get('scaler', None)
        svm_features = svm_data.get('features', [])
        print(f"SVM model extracted from dictionary. Type: {type(svm_model)}")
    else:
        # Si ce n'est pas un dictionnaire, c'est peut-être directement le modèle
        svm_model = svm_data
        svm_scaler = None
        svm_features = []
        print(f"SVM model loaded directly. Type: {type(svm_model)}")
    
    # Verify it's a proper model
    if not hasattr(svm_model, 'predict'):
        print(f"ERROR: SVM model doesn't have predict method! Type: {type(svm_model)}")
        print("Available attributes:", dir(svm_model))
        exit()
    
    print("SVM model loaded successfully with predict method!")
    
except Exception as e:
    print(f"Error loading SVM model: {e}")
    exit()

# MySQL Connection
try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="PreeclampsiaRiskDB"
    )
    cursor = db.cursor()
except Exception as e:
    print(f"MySQL connection error: {e}")
    exit()

# Tkinter Interface
root = tk.Tk()
root.title("Preeclampsia Risk Assessment (Ensemble Method: RF + SVM)")
root.geometry("500x700")

def calculate_bmi():
    """Automatically calculates BMI when height or weight changes"""
    try:
        height_str = entry_height.get().strip()
        weight_str = entry_weight.get().strip()
        
        if not height_str or not weight_str:
            # If one field is empty, clear BMI
            entry_bmi.config(state="normal")
            entry_bmi.delete(0, tk.END)
            entry_bmi.config(state="readonly")
            bmi_status.config(text="", fg="black")
            return
            
        height = float(height_str)
        weight = float(weight_str)
        
        if height > 0 and weight > 0:
            if height > 3:  # If height in cm, convert to m
                height = height / 100
            bmi = weight / (height ** 2)
            
            # Update BMI field
            entry_bmi.config(state="normal")
            entry_bmi.delete(0, tk.END)
            entry_bmi.insert(0, f"{bmi:.1f}")
            entry_bmi.config(state="readonly")
            
            # BMI status indication
            if bmi < 18.5:
                bmi_status.config(text="(Underweight)", fg="blue")
            elif bmi < 25:
                bmi_status.config(text="(Normal)", fg="green")
            elif bmi < 30:
                bmi_status.config(text="(Overweight)", fg="orange")
            else:
                bmi_status.config(text="(Obese)", fg="red")
        else:
            bmi_status.config(text="Invalid values", fg="red")
    except ValueError:
        entry_bmi.config(state="normal")
        entry_bmi.delete(0, tk.END)
        entry_bmi.config(state="readonly")
        bmi_status.config(text="Invalid values", fg="red")

def update_weeks_status():
    """Automatically updates after_20_weeks based on gestational week"""
    try:
        week_str = entry_week.get().strip()
        if not week_str:
            var_after_20.set(0)
            week_status.config(text="", fg="black")
            return
            
        week = int(week_str)
        if week >= 20:
            var_after_20.set(1)
            week_status.config(text="(≥20 weeks)", fg="red")
        else:
            var_after_20.set(0)
            week_status.config(text="(<20 weeks)", fg="green")
    except ValueError:
        var_after_20.set(0)
        week_status.config(text="Invalid value", fg="red")

def save_data(age, sbp, dbp, BMI, multiple, diabetes, first_pregnancy, 
              after_20_weeks, chronic_hypertension, Proteinuria, pe_value):
    """Saves data to MySQL database"""
    try:
        sql = """
        INSERT INTO pregnancy_cases (age, sbp, dbp, BMI, multiple, diabetes,
        first_pregnancy, after_20_weeks, chronic_hypertension, Proteinuria, PE)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        values = (age, sbp, dbp, BMI, multiple, diabetes,
                  first_pregnancy, after_20_weeks, chronic_hypertension,
                  Proteinuria, pe_value)
        
        cursor.execute(sql, values)
        db.commit()
        messagebox.showinfo("Success", "Data saved successfully to database.")
        
    except Exception as e:
        messagebox.showerror("Save Error", f"Error saving data: {str(e)}")

def ensemble_prediction(features):
    """
    Ensemble method combining Random Forest and SVM predictions
    Returns: ensemble_prediction, ensemble_probability, individual_results
    """
    try:
        print(f"Input features shape: {features.shape}")
        print(f"Input features: {features}")
        print(f"RF Model type: {type(rf_model)}")
        print(f"SVM Model type: {type(svm_model)}")
        
        # Verify models are loaded correctly
        if not hasattr(rf_model, 'predict'):
            raise AttributeError("Random Forest model is not a valid sklearn model")
        if not hasattr(svm_model, 'predict'):
            raise AttributeError("SVM model is not a valid sklearn model")
        
        # Préparation des données pour SVM (normalisation si nécessaire)
        features_for_svm = features.copy()
        
        # Si nous avons un scaler pour SVM, l'appliquer
        if svm_scaler is not None:
            print("Applying SVM scaler...")
            # Créer un DataFrame pour faciliter la manipulation
            feature_names = ['age', 'sbp', 'dbp', 'BMI', 'multiple', 'diabetes',
                           'first_pregnancy', 'after_20_weeks', 'chronic_hypertension', 'Proteinuria']
            df_features = pd.DataFrame(features, columns=feature_names)
            
            # Normaliser les caractéristiques continues
            cont_features = ['age', 'sbp', 'dbp', 'BMI']
            df_features[cont_features] = svm_scaler.transform(df_features[cont_features])
            features_for_svm = df_features.values
        
        # Random Forest prediction
        print("Making Random Forest prediction...")
        rf_prediction = rf_model.predict(features)[0]
        print(f"RF prediction: {rf_prediction}")
        
        if hasattr(rf_model, 'predict_proba'):
            rf_probability = rf_model.predict_proba(features)[0]
            print(f"RF probability: {rf_probability}")
        else:
            # If RF doesn't have predict_proba, create binary probabilities
            rf_probability = [1-rf_prediction, rf_prediction] if rf_prediction == 1 else [1, 0]
            print(f"RF probability (binary): {rf_probability}")
        
        # SVM prediction
        print("Making SVM prediction...")
        svm_prediction = svm_model.predict(features_for_svm)[0]
        print(f"SVM prediction: {svm_prediction}")
        
        # Check if SVM has predict_proba method
        if hasattr(svm_model, 'predict_proba'):
            svm_probability = svm_model.predict_proba(features_for_svm)[0]
            print(f"SVM probability: {svm_probability}")
        else:
            # If SVM doesn't have predict_proba, try decision_function
            if hasattr(svm_model, 'decision_function'):
                svm_decision = svm_model.decision_function(features_for_svm)[0]
                print(f"SVM decision function: {svm_decision}")
                # Convert decision function to probability using sigmoid
                try:
                    from scipy.special import expit
                    svm_prob_positive = expit(svm_decision)
                except ImportError:
                    # Fallback sigmoid implementation
                    svm_prob_positive = 1 / (1 + np.exp(-svm_decision))
                svm_probability = [1 - svm_prob_positive, svm_prob_positive]
                print(f"SVM probability (from decision): {svm_probability}")
            else:
                # If no probability method available, use binary
                svm_probability = [1-svm_prediction, svm_prediction] if svm_prediction == 1 else [1, 0]
                print(f"SVM probability (binary): {svm_probability}")
        
        # Ensure probabilities are in correct format
        if len(rf_probability) != 2:
            rf_probability = [1-rf_prediction, rf_prediction]
        if len(svm_probability) != 2:
            svm_probability = [1-svm_prediction, svm_prediction]
        
        # Ensemble method: Average of probabilities
        ensemble_prob_negative = (rf_probability[0] + svm_probability[0]) / 2
        ensemble_prob_positive = (rf_probability[1] + svm_probability[1]) / 2
        ensemble_probability = [ensemble_prob_negative, ensemble_prob_positive]
        
        print(f"Ensemble probability: {ensemble_probability}")
        
        # Final prediction based on ensemble probability
        ensemble_prediction = 1 if ensemble_prob_positive > 0.5 else 0
        
        # Store individual results for display
        individual_results = {
            'rf_prediction': int(rf_prediction),
            'rf_probability': float(rf_probability[1]),
            'svm_prediction': int(svm_prediction),
            'svm_probability': float(svm_probability[1]),
            'ensemble_probability': float(ensemble_prob_positive)
        }
        
        print(f"Final individual results: {individual_results}")
        
        return ensemble_prediction, ensemble_probability, individual_results
        
    except Exception as e:
        print(f"Detailed error in ensemble prediction: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise e

# Main prediction function
def predict_risk():
    try:
        # Validation and data retrieval from form
        age_str = entry_age.get().strip()
        sbp_str = entry_sbp.get().strip()
        dbp_str = entry_dbp.get().strip()
        bmi_str = entry_bmi.get().strip()
        week_str = entry_week.get().strip()
        
        # Check that all required fields are filled
        if not all([age_str, sbp_str, dbp_str, bmi_str, week_str]):
            messagebox.showerror("Error", "Please fill all required fields:\n- Age\n- Gestational week\n- Height and Weight (to calculate BMI)\n- Blood pressures")
            return
        
        # Conversion with specific error messages
        try:
            age = int(age_str)
            if age < 15 or age > 50:
                messagebox.showwarning("Warning", "Unusual age (15-50 years expected)")
        except ValueError:
            messagebox.showerror("Error", f"Invalid age: '{age_str}'. Enter a whole number.")
            return
            
        try:
            sbp = float(sbp_str)
            if sbp < 80 or sbp > 200:
                messagebox.showwarning("Warning", "Unusual systolic pressure (80-200 mmHg expected)")
        except ValueError:
            messagebox.showerror("Error", f"Invalid systolic pressure: '{sbp_str}'. Enter a number.")
            return
            
        try:
            dbp = float(dbp_str)
            if dbp < 50 or dbp > 120:
                messagebox.showwarning("Warning", "Unusual diastolic pressure (50-120 mmHg expected)")
        except ValueError:
            messagebox.showerror("Error", f"Invalid diastolic pressure: '{dbp_str}'. Enter a number.")
            return
            
        try:
            BMI = float(bmi_str)
            if BMI < 15 or BMI > 50:
                messagebox.showwarning("Warning", "Unusual BMI (15-50 expected)")
        except ValueError:
            messagebox.showerror("Error", f"Invalid BMI: '{bmi_str}'. Check height and weight.")
            return
            
        try:
            week = int(week_str)
            if week < 1 or week > 42:
                messagebox.showwarning("Warning", "Unusual gestational week (1-42 expected)")
        except ValueError:
            messagebox.showerror("Error", f"Invalid gestational week: '{week_str}'. Enter a whole number.")
            return
        
        # Automatic calculation of after_20_weeks
        after_20_weeks = 1 if week >= 20 else 0
        
        diabetes = var_diabetes.get()
        first_pregnancy = var_first_pregnancy.get()
        multiple = var_multiple.get()
        chronic_hypertension = var_chronic_htn.get()
        Proteinuria = var_proteinuria.get()

        # Data preparation for models
        features = np.array([[age, sbp, dbp, BMI, multiple, diabetes,
                              first_pregnancy, after_20_weeks, chronic_hypertension, Proteinuria]])
        
        print(f"Input data: {features}")  # Debug

        # Ensemble prediction
        ensemble_pred, ensemble_prob, individual_results = ensemble_prediction(features)
        
        print(f"Individual results: {individual_results}")  # Debug
        
        # Classification by risk levels with 3 categories using ensemble probability
        risk_high_prob = individual_results['ensemble_probability']
        
        # Determine risk level based on probability thresholds
        if risk_high_prob < 0.40:  # Less than 40%
            risk_level = "LOW RISK"
            risk_color = "green"
            pe_value = 0
        elif risk_high_prob <= 0.65:  # Between 40% and 65%
            risk_level = "MEDIUM RISK"
            risk_color = "orange"
            pe_value = 1
        else:  # Greater than 65%
            risk_level = "HIGH RISK"
            risk_color = "red"
            pe_value = 1
        
        # Display detailed result
        prob_high = risk_high_prob * 100
        rf_prob = individual_results['rf_probability'] * 100
        svm_prob = individual_results['svm_probability'] * 100
        
        result_text = f"ENSEMBLE RESULT: {risk_level}\n\n"
        result_text += f"Final Risk Probability: {prob_high:.1f}%\n\n"
        result_text += f"Individual Model Results:\n"
        result_text += f"• Random Forest: {rf_prob:.1f}%\n"
        result_text += f"• SVM: {svm_prob:.1f}%\n\n"
        result_text += f"Gestational week: {week}"
        
        # Color based on risk level
        result_label.config(text=result_text, fg=risk_color)

        # Ask for confirmation to save
        response = messagebox.askyesno(
            "Save Data", 
            f"Ensemble Assessment Result:\n{risk_level}\n\n"
            f"Final Risk Probability: {prob_high:.1f}%\n"
            f"Random Forest: {rf_prob:.1f}%\n"
            f"SVM: {svm_prob:.1f}%\n\n"
            f"Do you want to save this data to the database?"
        )
        
        if response:  # If user clicks "Yes"
            save_data(age, sbp, dbp, BMI, multiple, diabetes,
                      first_pregnancy, after_20_weeks, chronic_hypertension,
                      Proteinuria, pe_value)
        else:
            messagebox.showinfo("Information", "Data was not saved.")
        
    except ValueError as e:
        messagebox.showerror("Validation Error", "Please check that all numeric fields contain valid values.")
        print(f"Validation error: {e}")  # Debug
    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error: {str(e)}")
        print(f"Detailed error: {e}")  # Debug

def clear_form():
    """Function to clear the form"""
    entry_age.delete(0, tk.END)
    entry_sbp.delete(0, tk.END)
    entry_dbp.delete(0, tk.END)
    entry_height.delete(0, tk.END)
    entry_weight.delete(0, tk.END)
    entry_bmi.config(state="normal")
    entry_bmi.delete(0, tk.END)
    entry_bmi.config(state="readonly")
    entry_week.delete(0, tk.END)
    var_after_20.set(0)
    var_diabetes.set(0)
    var_first_pregnancy.set(0)
    var_multiple.set(0)
    var_chronic_htn.set(0)
    var_proteinuria.set(0)
    result_label.config(text="")
    bmi_status.config(text="")
    week_status.config(text="")

# Enhanced user interface
title_label = tk.Label(root, text="Preeclampsia Risk Assessment", 
                      font=("Arial", 14, "bold"))
title_label.grid(row=0, column=0, columnspan=3, pady=10)

subtitle_label = tk.Label(root, text="Ensemble Method: Random Forest + SVM", 
                         font=("Arial", 10, "italic"), fg="blue")
subtitle_label.grid(row=1, column=0, columnspan=3, pady=5)

# Basic information
tk.Label(root, text="Basic Information:", font=("Arial", 10, "bold")).grid(row=2, column=0, columnspan=3, pady=(10,5))

tk.Label(root, text="Age (years):").grid(row=3, column=0, sticky="w", padx=10)
entry_age = tk.Entry(root, width=15)
entry_age.grid(row=3, column=1, padx=5, pady=2)

tk.Label(root, text="Gestational week:").grid(row=4, column=0, sticky="w", padx=10)
entry_week = tk.Entry(root, width=15)
entry_week.grid(row=4, column=1, padx=5, pady=2)
entry_week.bind('<KeyRelease>', lambda e: update_weeks_status())
week_status = tk.Label(root, text="", font=("Arial", 8))
week_status.grid(row=4, column=2, padx=5)

# Anthropometry
tk.Label(root, text="Anthropometry:", font=("Arial", 10, "bold")).grid(row=5, column=0, columnspan=3, pady=(10,5))

tk.Label(root, text="Height (cm or m):").grid(row=6, column=0, sticky="w", padx=10)
entry_height = tk.Entry(root, width=15)
entry_height.grid(row=6, column=1, padx=5, pady=2)
entry_height.bind('<KeyRelease>', lambda e: calculate_bmi())

tk.Label(root, text="Weight (kg):").grid(row=7, column=0, sticky="w", padx=10)
entry_weight = tk.Entry(root, width=15)
entry_weight.grid(row=7, column=1, padx=5, pady=2)
entry_weight.bind('<KeyRelease>', lambda e: calculate_bmi())

tk.Label(root, text="BMI (calculated):").grid(row=8, column=0, sticky="w", padx=10)
entry_bmi = tk.Entry(root, width=15, state="normal", bg="lightgray")
entry_bmi.grid(row=8, column=1, padx=5, pady=2)
bmi_status = tk.Label(root, text="", font=("Arial", 8))
bmi_status.grid(row=8, column=2, padx=5)

# Blood pressure
tk.Label(root, text="Blood Pressure:", font=("Arial", 10, "bold")).grid(row=9, column=0, columnspan=3, pady=(10,5))

tk.Label(root, text="Systolic pressure (mmHg):").grid(row=10, column=0, sticky="w", padx=10)
entry_sbp = tk.Entry(root, width=15)
entry_sbp.grid(row=10, column=1, padx=5, pady=2)

tk.Label(root, text="Diastolic pressure (mmHg):").grid(row=11, column=0, sticky="w", padx=10)
entry_dbp = tk.Entry(root, width=15)
entry_dbp.grid(row=11, column=1, padx=5, pady=2)

# Risk factors
tk.Label(root, text="Risk Factors:", font=("Arial", 10, "bold")).grid(row=12, column=0, columnspan=3, pady=(10,5))

var_after_20 = tk.IntVar()
tk.Checkbutton(root, text="Pregnancy ≥ 20 weeks (auto)", 
              variable=var_after_20, state="disabled").grid(row=13, column=0, columnspan=3, sticky="w", padx=10)

var_diabetes = tk.IntVar()
tk.Checkbutton(root, text="Diabetes", variable=var_diabetes).grid(row=14, column=0, columnspan=3, sticky="w", padx=10)

var_first_pregnancy = tk.IntVar()
tk.Checkbutton(root, text="First pregnancy", variable=var_first_pregnancy).grid(row=15, column=0, columnspan=3, sticky="w", padx=10)

var_multiple = tk.IntVar()
tk.Checkbutton(root, text="Multiple pregnancy", variable=var_multiple).grid(row=16, column=0, columnspan=3, sticky="w", padx=10)

var_chronic_htn = tk.IntVar()
tk.Checkbutton(root, text="Chronic hypertension", variable=var_chronic_htn).grid(row=17, column=0, columnspan=3, sticky="w", padx=10)

var_proteinuria = tk.IntVar()
tk.Checkbutton(root, text="Proteinuria", variable=var_proteinuria).grid(row=18, column=0, columnspan=3, sticky="w", padx=10)

# Buttons
button_frame = tk.Frame(root)
button_frame.grid(row=19, column=0, columnspan=3, pady=20)

tk.Button(button_frame, text="Assess Risk", command=predict_risk, 
          bg="lightblue", font=("Arial", 11, "bold"), width=15).pack(side=tk.LEFT, padx=5)

tk.Button(button_frame, text="Clear Form", command=clear_form, 
          bg="lightgray", width=15).pack(side=tk.LEFT, padx=5)

# Result
result_label = tk.Label(root, text="", font=("Arial", 11, "bold"), wraplength=450, justify="left")
result_label.grid(row=20, column=0, columnspan=3, pady=15)

# Model information with updated risk levels
info_label = tk.Label(root, text="Ensemble Method (RF+SVM) - Risk levels: <40% (Low), 40-65% (Medium), >65% (High)", 
                     font=("Arial", 8), fg="gray")
info_label.grid(row=21, column=0, columnspan=3, pady=5)

# Proper MySQL connection closure
def on_closing():
    try:
        cursor.close()
        db.close()
    except:
        pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()