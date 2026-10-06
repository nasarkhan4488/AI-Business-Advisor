import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# AI BUSINESS ADVISOR - EXPANDED BUSINESS MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(BASE_DIR, "business_data.pkl")
MODEL_FILE = os.path.join(BASE_DIR, "business_model.pkl")


# ============================================================
# 1. EXPANDED BUSINESS DATA
# ============================================================

data = [

    # ================= UNIVERSITY TOWN =================

    ["University Town", "Cafe", 85000, 25, 45000, 7000, 1500000, 82, 1],
    ["University Town", "Fast Food", 85000, 22, 45000, 8500, 1200000, 86, 1],
    ["University Town", "Bakery", 85000, 18, 45000, 6500, 1100000, 84, 1],
    ["University Town", "Mini Mart", 85000, 20, 45000, 8000, 1300000, 83, 1],
    ["University Town", "Pharmacy", 85000, 15, 45000, 5000, 1000000, 80, 1],
    ["University Town", "Gym", 85000, 20, 45000, 6000, 2000000, 74, 0],
    ["University Town", "Mobile & Accessories", 85000, 16, 45000, 6500, 800000, 88, 1],
    ["University Town", "Computer & IT Shop", 85000, 12, 45000, 5000, 900000, 87, 1],
    ["University Town", "Laundry Service", 85000, 10, 45000, 4000, 700000, 89, 1],
    ["University Town", "Study / Co-working Space", 85000, 8, 45000, 3500, 1400000, 85, 1],
    ["University Town", "Car Wash", 85000, 12, 45000, 3000, 900000, 76, 0],
    ["University Town", "Courier Point", 85000, 8, 45000, 4500, 600000, 91, 1],


    # ================= HAYATABAD =================

    ["Hayatabad", "Cafe", 120000, 40, 60000, 9500, 1800000, 78, 1],
    ["Hayatabad", "Fast Food", 120000, 35, 60000, 10500, 1600000, 82, 1],
    ["Hayatabad", "Bakery", 120000, 25, 60000, 8000, 1500000, 80, 1],
    ["Hayatabad", "Mini Mart", 120000, 30, 60000, 10000, 1700000, 79, 1],
    ["Hayatabad", "Pharmacy", 120000, 25, 60000, 7000, 1200000, 81, 1],
    ["Hayatabad", "Gym", 120000, 30, 60000, 7500, 2500000, 70, 0],
    ["Hayatabad", "Mobile & Accessories", 120000, 20, 60000, 7500, 900000, 85, 1],
    ["Hayatabad", "Computer & IT Shop", 120000, 15, 60000, 5500, 1000000, 84, 1],
    ["Hayatabad", "Laundry Service", 120000, 12, 60000, 4500, 800000, 87, 1],
    ["Hayatabad", "Study / Co-working Space", 120000, 10, 60000, 5000, 1500000, 82, 1],
    ["Hayatabad", "Car Wash", 120000, 15, 60000, 4000, 1000000, 78, 1],
    ["Hayatabad", "Courier Point", 120000, 10, 60000, 5000, 650000, 88, 1],


    # ================= SADDAR =================

    ["Saddar", "Cafe", 150000, 70, 80000, 12000, 2500000, 65, 0],
    ["Saddar", "Fast Food", 150000, 60, 80000, 14000, 2200000, 70, 1],
    ["Saddar", "Bakery", 150000, 45, 80000, 10000, 2000000, 68, 0],
    ["Saddar", "Mini Mart", 150000, 50, 80000, 13000, 2300000, 67, 0],
    ["Saddar", "Pharmacy", 150000, 50, 80000, 9000, 1500000, 72, 1],
    ["Saddar", "Gym", 150000, 60, 80000, 10000, 3000000, 60, 0],
    ["Saddar", "Mobile & Accessories", 150000, 40, 80000, 11000, 1000000, 78, 1],
    ["Saddar", "Computer & IT Shop", 150000, 30, 80000, 7000, 1200000, 76, 1],
    ["Saddar", "Laundry Service", 150000, 20, 80000, 5000, 900000, 80, 1],
    ["Saddar", "Study / Co-working Space", 150000, 25, 80000, 5000, 1800000, 70, 0],
    ["Saddar", "Car Wash", 150000, 25, 80000, 5000, 1200000, 74, 1],
    ["Saddar", "Courier Point", 150000, 15, 80000, 6000, 700000, 84, 1],


    # ================= DHA =================

    ["DHA", "Cafe", 90000, 20, 70000, 8500, 2200000, 78, 1],
    ["DHA", "Fast Food", 90000, 18, 70000, 9000, 2000000, 81, 1],
    ["DHA", "Bakery", 90000, 15, 70000, 7000, 1800000, 82, 1],
    ["DHA", "Mini Mart", 90000, 18, 70000, 8500, 1900000, 79, 1],
    ["DHA", "Pharmacy", 90000, 18, 70000, 6000, 1300000, 80, 1],
    ["DHA", "Gym", 90000, 25, 70000, 7000, 2800000, 72, 0],
    ["DHA", "Mobile & Accessories", 90000, 12, 70000, 6000, 1000000, 84, 1],
    ["DHA", "Computer & IT Shop", 90000, 10, 70000, 4500, 1200000, 82, 1],
    ["DHA", "Laundry Service", 90000, 8, 70000, 4000, 850000, 88, 1],
    ["DHA", "Study / Co-working Space", 90000, 8, 70000, 4500, 1600000, 84, 1],
    ["DHA", "Car Wash", 90000, 12, 70000, 3500, 1100000, 80, 1],
    ["DHA", "Courier Point", 90000, 8, 70000, 4500, 650000, 90, 1],


    # ================= RING ROAD =================

    ["Ring Road", "Cafe", 180000, 35, 40000, 11000, 1200000, 83, 1],
    ["Ring Road", "Fast Food", 180000, 28, 40000, 12500, 1000000, 87, 1],
    ["Ring Road", "Bakery", 180000, 22, 40000, 9000, 900000, 85, 1],
    ["Ring Road", "Mini Mart", 180000, 25, 40000, 14000, 1100000, 88, 1],
    ["Ring Road", "Pharmacy", 180000, 22, 40000, 8000, 900000, 84, 1],
    ["Ring Road", "Gym", 180000, 28, 40000, 9000, 1800000, 78, 1],
    ["Ring Road", "Mobile & Accessories", 180000, 20, 40000, 10000, 750000, 90, 1],
    ["Ring Road", "Computer & IT Shop", 180000, 15, 40000, 6500, 850000, 88, 1],
    ["Ring Road", "Laundry Service", 180000, 12, 40000, 5500, 650000, 92, 1],
    ["Ring Road", "Study / Co-working Space", 180000, 10, 40000, 4000, 1200000, 82, 1],
    ["Ring Road", "Car Wash", 180000, 18, 40000, 6000, 850000, 86, 1],
    ["Ring Road", "Courier Point", 180000, 10, 40000, 6500, 550000, 93, 1],
]


columns = [
    "Area",
    "Business_Type",
    "Population",
    "Competition",
    "Monthly_Rent",
    "Customers",
    "Budget_Required",
    "Opportunity_Score",
    "Suitable"
]

df = pd.DataFrame(data, columns=columns)


# ============================================================
# 2. ADD LOCATION FEATURES
# ============================================================

location_features = {
    "University Town": {
        "Population_Density": 8500,
        "Urban_Proportion": 85,
        "Annual_Growth_Rate": 3.2
    },
    "Hayatabad": {
        "Population_Density": 7200,
        "Urban_Proportion": 90,
        "Annual_Growth_Rate": 3.0
    },
    "Saddar": {
        "Population_Density": 9500,
        "Urban_Proportion": 95,
        "Annual_Growth_Rate": 2.5
    },
    "DHA": {
        "Population_Density": 6000,
        "Urban_Proportion": 92,
        "Annual_Growth_Rate": 3.5
    },
    "Ring Road": {
        "Population_Density": 7000,
        "Urban_Proportion": 80,
        "Annual_Growth_Rate": 3.8
    }
}

df["Population_Density"] = df["Area"].map(
    lambda x: location_features[x]["Population_Density"]
)

df["Urban_Proportion"] = df["Area"].map(
    lambda x: location_features[x]["Urban_Proportion"]
)

df["Annual_Growth_Rate"] = df["Area"].map(
    lambda x: location_features[x]["Annual_Growth_Rate"]
)


# ============================================================
# 3. ONE-HOT ENCODE BUSINESS TYPES
# ============================================================

df = pd.get_dummies(
    df,
    columns=["Business_Type"],
    prefix="Business_Type"
)


# ============================================================
# 4. PREPARE FEATURES
# ============================================================

feature_columns = [
    "Population",
    "Population_Density",
    "Urban_Proportion",
    "Annual_Growth_Rate",
    "Competition",
    "Monthly_Rent",
    "Customers",
    "Budget_Required"
]

business_columns = [
    col for col in df.columns
    if col.startswith("Business_Type_")
]

feature_columns += business_columns

X = df[feature_columns]
y = df["Suitable"]


# ============================================================
# 5. TRAIN MODEL
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# ============================================================
# 6. TEST MODEL
# ============================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print()
print("=" * 60)
print("AI BUSINESS ADVISOR MODEL TRAINING")
print("=" * 60)

print(f"Dataset rows: {len(df)}")
print(f"Business categories: {len(business_columns)}")
print(f"Model accuracy on test split: {accuracy:.2%}")

print()
print("Business Categories:")
for col in business_columns:
    print(" -", col.replace("Business_Type_", ""))

print()
print("Classification Report:")
print(classification_report(y_test, predictions))


# ============================================================
# 7. SAVE MODEL + DATA
# ============================================================

joblib.dump(model, MODEL_FILE)

# Save original-style dataframe before one-hot encoding
original_df = pd.DataFrame(data, columns=columns)

joblib.dump(original_df, DATA_FILE)


print()
print("Model saved:")
print(MODEL_FILE)

print()
print("Business data saved:")
print(DATA_FILE)

print()
print("Training completed successfully!")
print("=" * 60)