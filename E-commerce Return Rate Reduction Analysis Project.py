import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load the data
df = pd.read_csv(r'C:\Users\sarik\OneDrive\INTERN\ecommerce_returns (1).csv')

# Clean column names
df.columns = df.columns.str.strip()
print("Columns:", df.columns.tolist())

# Label encode categorical columns (only if they exist)
categoricals = ['Category', 'Supplier', 'Region', 'Channel']
le = LabelEncoder()
for col in categoricals:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])
    else:
        print(f"⚠️ Skipping missing column: {col}")

# Check if 'Returned' column exists
if 'Returned' not in df.columns:
    raise ValueError("❌ 'Returned' column is missing in your dataset.")

# Define features and target
X = df.drop(['OrderID', 'ProductID', 'Returned'], axis=1)
y = df['Returned']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))

# Predict return probability
df['return_probability'] = model.predict_proba(X)[:, 1]

# Save high-risk products
high_risk = df[df['return_probability'] > 0.6]
high_risk.to_csv("high_risk_products.csv", index=False)
print("✅ High-risk products saved to 'high_risk_products.csv'")

