import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# load dataset
df = pd.read_csv("insurance_claims.csv")

# drop unused columns
df = df.drop(columns=["_c39","policy_bind_date","incident_date"])

# convert target column
df["fraud_reported"] = df["fraud_reported"].map({"Y":1,"N":0})

# encode incident_severity (categorical)
le = LabelEncoder()
df["incident_severity"] = le.fit_transform(df["incident_severity"])

# choose 6 clean features
features = [
    "age",
    "policy_deductable",
    "policy_annual_premium",
    "umbrella_limit",
    "incident_severity",
    "total_claim_amount"
]

X = df[features]
y = df["fraud_reported"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ML pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100))
])

pipeline.fit(X_train, y_train)

# save model
with open("fraud_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model trained and saved successfully!")