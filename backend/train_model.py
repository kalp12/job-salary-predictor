import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample dataset (Replace with real salary dataset)
data = {
    "experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "salary": [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000]
}

df = pd.DataFrame(data)

# Train-test split
X = df[["experience"]]
y = df["salary"]

model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, "model.pkl")
print("Model trained and saved!")