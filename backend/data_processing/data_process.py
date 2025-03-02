import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("../Salary_Data.csv")

# Drop missing values
df.dropna(inplace=True)

# Define numeric and categorical features
numeric_features = ["Age", "experience"]
categorical_features = ["Gender", "Education Level", "Job Title"]
target_column = "Salary"

# Scale numeric features FIRST
feature_scaler = StandardScaler()
df[numeric_features] = feature_scaler.fit_transform(df[numeric_features])

# Scale Salary separately
salary_scaler = StandardScaler()
df[target_column] = salary_scaler.fit_transform(df[[target_column]])

# Apply one-hot encoding SECOND
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Split data
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_split=10, random_state=42)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = salary_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0]  # Convert back to original scale
    y_test_original = salary_scaler.inverse_transform(y_test.values.reshape(-1, 1))[:, 0]
    
    mae = mean_absolute_error(y_test_original, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
    r2 = r2_score(y_test_original, y_pred)
    
    return {"MAE": mae, "RMSE": rmse, "R2 Score": r2}

rf_metrics = evaluate_model(rf_model, X_test, y_test)
gb_metrics = evaluate_model(gb_model, X_test, y_test)

print("Random Forest Metrics:", rf_metrics)
print("Gradient Boosting Metrics:", gb_metrics)

# Prediction Function
def predict_salary(age, gender, education, job_title, experience, model):
    x = np.zeros(len(X.columns))  # Create an empty input array

    # Scale numeric featuresf
    scaled_features = feature_scaler.transform([[age, experience]])[0]
    x[0:2] = scaled_features  # Assign scaled Age & Experience

    # One-hot encode categorical variables
    for col_name in [f"Gender_{gender}", f"Education Level_{education}", f"Job Title_{job_title}"]:
        if col_name in X.columns:
            x[np.where(X.columns == col_name)[0][0]] = 1

    # Convert input to DataFrame
    x_df = pd.DataFrame([x], columns=X.columns)

    # Predict scaled salary
    predicted_salary_scaled = model.predict(x_df)[0]

    # Convert back to original salary scale
    predicted_salary = salary_scaler.inverse_transform([[predicted_salary_scaled]])[0][0]

    return predicted_salary

# Example prediction
predicted_salary = predict_salary(32, "Male", "Bachelor's", "software engineer", 5.0, rf_model)
print(f"Predicted Salary: ${predicted_salary:.2f}")
import joblib
# joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(gb_model, "gb_model.pkl")
joblib.dump(feature_scaler, "feature_scaler.pkl")
joblib.dump(salary_scaler, "salary_scaler.pkl")
joblib.dump(X, "X_columns.pkl")

predicted_salary = predict_salary(32, "Male", "Bachelor's", "software engineer", 5.0, gb_model)
print(f"Predicted Salary: ${predicted_salary:.2f}")