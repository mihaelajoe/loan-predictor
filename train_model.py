import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data from the correct sheet (replace with your actual sheet name)
df = pd.read_excel("loan.xlsx", sheet_name="Data")  # replace "data" with your sheet name if different

# Drop missing values
df = df.dropna()

# Drop columns that won't help the model
df = df.drop(columns=["ID", "ZIP Code"])

# Features and target
X = df.drop("Personal Loan", axis=1)
y = df["Personal Loan"]

# If 'Personal Loan' is not numeric, convert it here (but likely already numeric 0/1)
# y = y.map({ 'Yes': 1, 'No': 0 })  # only if needed

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "loan_model.pkl")

print("✅ Model trained and saved as loan_model.pkl")
