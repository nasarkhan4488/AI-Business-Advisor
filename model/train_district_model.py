import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUSINESS_DATA = os.path.join(
    BASE_DIR,
    "backend",
    "business_data.pkl"
)

CENSUS_DATA = os.path.join(
    BASE_DIR,
    "data",
    "population",
    "kp_district_population_2023.csv"
)

MODEL_OUTPUT = os.path.join(
    BASE_DIR,
    "backend",
    "business_model.pkl"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("AI BUSINESS ADVISOR - MODEL TRAINING")
print("=" * 70)

print("\nLoading business data...")

business_df = joblib.load(BUSINESS_DATA)

print(f"Business records: {len(business_df)}")

print("\nLoading KPK Census 2023 data...")

census_df = pd.read_csv(CENSUS_DATA)

print(f"KPK districts: {len(census_df)}")


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

business_df.columns = business_df.columns.str.strip()
census_df.columns = census_df.columns.str.strip()


# ============================================================
# 4. STANDARDIZE DISTRICT NAMES
# ============================================================

census_df["district"] = (
    census_df["district"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# 5. CREATE DISTRICT CONTEXT
# ============================================================

print("\nPreparing district information...")

print("\nAvailable Census columns:")
print(census_df.columns.tolist())


# ============================================================
# 6. CREATE TRAINING DATA
# ============================================================

print("\nPreparing training data...")

training_rows = []

business_types = business_df["Business_Type"].unique()

print("\nBusiness types:")
print(list(business_types))


# ------------------------------------------------------------
# IMPORTANT:
# Existing business records contain the relationship between
# business characteristics and Suitable.
#
# Census information is added as district context.
#
# Since the existing 15 records do not contain a district,
# we use their original area/business information as seed
# examples and attach representative KPK district contexts.
# ------------------------------------------------------------

for _, business in business_df.iterrows():

    for _, district in census_df.iterrows():

        row = {
            "Business_Type": business["Business_Type"],
            "Population": district["population_2023"],
            "Population_Density": district["population_density"],
            "Urban_Proportion": district["urban_proportion"],
            "Annual_Growth_Rate": district["annual_growth_rate"],
            "Competition": business["Competition"],
            "Monthly_Rent": business["Monthly_Rent"],
            "Customers": business["Customers"],
            "Budget_Required": business["Budget_Required"],
            "Suitable": business["Suitable"],
        }

        training_rows.append(row)


training_df = pd.DataFrame(training_rows)

print(f"\nGenerated training scenarios: {len(training_df)}")


# ============================================================
# 7. CREATE BUSINESS TYPE ONE-HOT COLUMNS
# ============================================================

training_df = pd.get_dummies(
    training_df,
    columns=["Business_Type"],
    prefix="Business_Type"
)


# ============================================================
# 8. REQUIRED MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "Population",
    "Population_Density",
    "Urban_Proportion",
    "Annual_Growth_Rate",
    "Competition",
    "Monthly_Rent",
    "Customers",
    "Budget_Required",

    "Business_Type_Cafe",
    "Business_Type_Gym",
    "Business_Type_Pharmacy",
]


# Add missing business type columns

for feature in MODEL_FEATURES:

    if feature not in training_df.columns:
        training_df[feature] = 0


# ============================================================
# 9. PREPARE X AND Y
# ============================================================

X = training_df[MODEL_FEATURES]

y = training_df["Suitable"].astype(int)


print("\nFeature columns:")
for feature in MODEL_FEATURES:
    print(" -", feature)


print("\nTarget distribution:")
print(y.value_counts().sort_index())


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


model.fit(X, y)


# ============================================================
# 11. CROSS VALIDATION
# ============================================================

print("\nRunning cross-validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


print("\nCross-validation accuracy:")
print(scores)

print(
    f"\nAverage CV Accuracy: {scores.mean() * 100:.2f}%"
)

print(
    f"CV Standard Deviation: {scores.std() * 100:.2f}%"
)


# ============================================================
# 12. TRAINING REPORT
# ============================================================

predictions = model.predict(X)

print("\nTraining classification report:")
print(
    classification_report(
        y,
        predictions,
        zero_division=0
    )
)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y,
        predictions
    )
)


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "Feature": MODEL_FEATURES,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


print("\nFeature Importance:")
print(
    importance_df.to_string(index=False)
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_OUTPUT
)


print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print(f"\nModel saved to:")
print(MODEL_OUTPUT)

print("\nModel features:")
print(MODEL_FEATURES)

print("\nIMPORTANT:")
print(
    "This is a prototype decision-support model. "
    "The expanded scenarios combine existing business examples "
    "with Census district context and should not be interpreted "
    "as real-world business outcome data."
)

print("=" * 70)