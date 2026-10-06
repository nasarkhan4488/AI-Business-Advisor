import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI Business Advisor",
    description="AI-powered business opportunity analysis for Khyber Pakhtunkhwa",
    version="3.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "backend",
    "business_model.pkl"
)

BUSINESS_DATA_PATH = os.path.join(
    BASE_DIR,
    "backend",
    "business_data.pkl"
)

CENSUS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "population",
    "kp_district_population_2023.csv"
)


# ============================================================
# LOAD MODEL + DATA
# ============================================================

print("Loading AI Business Advisor...")

model = joblib.load(MODEL_PATH)
business_df = joblib.load(BUSINESS_DATA_PATH)
census_df = pd.read_csv(CENSUS_PATH)

business_df.columns = business_df.columns.str.strip()
census_df.columns = census_df.columns.str.strip()

census_df["district"] = (
    census_df["district"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# MODEL FEATURES
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


# ============================================================
# REQUEST MODEL
# ============================================================

class BusinessRequest(BaseModel):
    area: str
    district: str
    budget: float
    latitude: float
    longitude: float


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_district_data(district_name: str):

    district_name = district_name.strip().upper()

    result = census_df[
        census_df["district"] == district_name
    ]

    if result.empty:
        return None

    return result.iloc[0]


def business_type_columns(df):

    df = df.copy()

    df["Business_Type"] = df["Business_Type"].astype(str)

    df = pd.get_dummies(
        df,
        columns=["Business_Type"],
        prefix="Business_Type"
    )

    for feature in MODEL_FEATURES:

        if feature not in df.columns:
            df[feature] = 0

    return df[MODEL_FEATURES]


def normalize(value, minimum, maximum):

    if maximum == minimum:
        return 50.0

    score = (
        (value - minimum)
        / (maximum - minimum)
    ) * 100

    return max(0, min(100, score))


# ============================================================
# DISTRICT DEMAND SCORE
# ============================================================

def calculate_district_demand(district):

    population_score = normalize(
        district["population_2023"],
        census_df["population_2023"].min(),
        census_df["population_2023"].max()
    )

    density_score = normalize(
        district["population_density"],
        census_df["population_density"].min(),
        census_df["population_density"].max()
    )

    urban_score = normalize(
        district["urban_proportion"],
        census_df["urban_proportion"].min(),
        census_df["urban_proportion"].max()
    )

    growth_score = normalize(
        district["annual_growth_rate"],
        census_df["annual_growth_rate"].min(),
        census_df["annual_growth_rate"].max()
    )

    demand_score = (
        population_score * 0.40
        + density_score * 0.25
        + urban_score * 0.20
        + growth_score * 0.15
    )

    return round(demand_score, 2)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI Business Advisor API is running",
        "version": "3.0",
        "districts_connected": len(census_df),
        "business_types": sorted(
            business_df["Business_Type"].unique().tolist()
        ),
        "model_features": MODEL_FEATURES,
    }


# ============================================================
# BUSINESSES
# ============================================================

@app.get("/businesses")
def get_businesses():

    return {
        "business_types": sorted(
            business_df["Business_Type"]
            .unique()
            .tolist()
        )
    }


# ============================================================
# AREAS
# ============================================================

@app.get("/areas")
def get_areas():

    return {
        "areas": sorted(
            business_df["Area"]
            .unique()
            .tolist()
        )
    }


# ============================================================
# DISTRICTS
# ============================================================

@app.get("/districts")
def get_districts():

    records = census_df.to_dict(
        orient="records"
    )

    return {
        "count": len(records),
        "districts": records
    }


# ============================================================
# SINGLE DISTRICT
# ============================================================

@app.get("/district/{district_name}")
def get_district(district_name: str):

    district = get_district_data(
        district_name
    )

    if district is None:

        raise HTTPException(
            status_code=404,
            detail="District not found"
        )

    return {
        "district": district["district"],
        "area_sq_km": float(
            district["area_sq_km"]
        ),
        "population_2023": int(
            district["population_2023"]
        ),
        "population_density": float(
            district["population_density"]
        ),
        "urban_proportion": float(
            district["urban_proportion"]
        ),
        "avg_household_size": float(
            district["avg_household_size"]
        ),
        "annual_growth_rate": float(
            district["annual_growth_rate"]
        ),
        "demand_score": calculate_district_demand(
            district
        ),
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict_business(request: BusinessRequest):

    # --------------------------------------------------------
    # Validate budget
    # --------------------------------------------------------

    if request.budget <= 0:

        raise HTTPException(
            status_code=400,
            detail="Budget must be greater than 0"
        )


    # --------------------------------------------------------
    # Get district
    # --------------------------------------------------------

    district = get_district_data(
        request.district
    )

    if district is None:

        raise HTTPException(
            status_code=404,
            detail=f"District '{request.district}' not found"
        )


    # --------------------------------------------------------
    # District information
    # --------------------------------------------------------

    population = float(
        district["population_2023"]
    )

    population_density = float(
        district["population_density"]
    )

    urban_proportion = float(
        district["urban_proportion"]
    )

    annual_growth_rate = float(
        district["annual_growth_rate"]
    )

    demand_score = calculate_district_demand(
        district
    )


    # --------------------------------------------------------
    # Business options
    # --------------------------------------------------------

    available_businesses = business_df[
        business_df["Budget_Required"]
        <= request.budget
    ].copy()


    if available_businesses.empty:

        raise HTTPException(
            status_code=400,
            detail=(
                "Your budget is lower than the "
                "minimum required budget for all "
                "available businesses."
            )
        )


    # --------------------------------------------------------
    # Prepare model input
    # --------------------------------------------------------

    model_input = available_businesses.copy()

    model_input["Population"] = population

    model_input["Population_Density"] = (
        population_density
    )

    model_input["Urban_Proportion"] = (
        urban_proportion
    )

    model_input["Annual_Growth_Rate"] = (
        annual_growth_rate
    )


    X = business_type_columns(
        model_input
    )


    # --------------------------------------------------------
    # ML predictions
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X
    )

    suitable_probability = (
        probabilities[:, 1] * 100
    )


    # --------------------------------------------------------
    # Score ranges
    # --------------------------------------------------------

    max_competition = max(
        float(business_df["Competition"].max()),
        1
    )

    max_rent = max(
        float(business_df["Monthly_Rent"].max()),
        1
    )

    max_customers = max(
        float(business_df["Customers"].max()),
        1
    )


    recommendations = []


    # --------------------------------------------------------
    # Calculate opportunity for every business
    # --------------------------------------------------------

    for index, (_, business) in enumerate(
        available_businesses.iterrows()
    ):

        competition = float(
            business["Competition"]
        )

        monthly_rent = float(
            business["Monthly_Rent"]
        )

        customers = float(
            business["Customers"]
        )

        required_budget = float(
            business["Budget_Required"]
        )


        # ----------------------------------------------------
        # Business factor scores
        # ----------------------------------------------------

        competition_score = (
            1 - competition / max_competition
        ) * 100

        rent_score = (
            1 - monthly_rent / max_rent
        ) * 100

        customer_score = (
            customers / max_customers
        ) * 100


        # ----------------------------------------------------
        # Business opportunity score
        # ----------------------------------------------------

        opportunity_score = (

            demand_score * 0.40

            + competition_score * 0.20

            + rent_score * 0.10

            + customer_score * 0.20

            + (
                suitable_probability[index]
                * 0.10
            )

        )


        opportunity_score = round(
            max(0, min(100, opportunity_score)),
            2
        )


        # ----------------------------------------------------
        # Budget compatibility
        # ----------------------------------------------------

        remaining_budget = (
            request.budget
            - required_budget
        )


        if request.budget >= required_budget * 1.5:

            budget_match = 100

            budget_label = "Excellent"

        elif request.budget >= required_budget:

            budget_match = 85

            budget_label = "Good"

        else:

            budget_match = 0

            budget_label = "Not Suitable"


        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        final_score = (

            suitable_probability[index] * 0.45

            + opportunity_score * 0.35

            + budget_match * 0.20

        )


        final_score = round(
            max(0, min(100, final_score)),
            2
        )


        # ----------------------------------------------------
        # Competition level
        # ----------------------------------------------------

        competition_ratio = (
            competition / max_competition
        )


        if competition_ratio <= 0.33:

            competition_level = "Low"

        elif competition_ratio <= 0.66:

            competition_level = "Medium"

        else:

            competition_level = "High"


        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        risk_score = competition_ratio * 100


        if final_score >= 75 and risk_score <= 40:

            risk_level = "Low"

        elif final_score >= 55 and risk_score <= 70:

            risk_level = "Medium"

        else:

            risk_level = "High"


        # ----------------------------------------------------
        # Suitability label
        # ----------------------------------------------------

        if suitable_probability[index] >= 75:

            suitability = "High"

        elif suitable_probability[index] >= 50:

            suitability = "Medium"

        else:

            suitability = "Low"


        # ----------------------------------------------------
        # Recommendation reasons
        # ----------------------------------------------------

        reasons = []


        if population >= census_df[
            "population_2023"
        ].median():

            reasons.append(
                "Strong population base"
            )


        if population_density >= census_df[
            "population_density"
        ].median():

            reasons.append(
                "Good population density"
            )


        if urban_proportion >= census_df[
            "urban_proportion"
        ].median():

            reasons.append(
                "Good urban market potential"
            )


        if annual_growth_rate >= census_df[
            "annual_growth_rate"
        ].median():

            reasons.append(
                "Positive population growth"
            )


        if competition_level == "Low":

            reasons.append(
                "Manageable competition"
            )


        if budget_label in ["Good", "Excellent"]:

            reasons.append(
                "Fits your budget"
            )


        if customers >= business_df[
            "Customers"
        ].median():

            reasons.append(
                "Good customer potential"
            )


        # Make sure there are reasons

        if not reasons:

            reasons.append(
                "Based on available business and district data"
            )


        recommendations.append({

            "Business_Type": business[
                "Business_Type"
            ],

            "Budget_Required": required_budget,

            "Suitable_Probability": round(
                float(
                    suitable_probability[index]
                ),
                2
            ),

            "Suitability": suitability,

            "Opportunity_Score": opportunity_score,

            "Final_Score": final_score,

            "Competition": competition,

            "Competition_Level": competition_level,

            "Monthly_Rent": monthly_rent,

            "Customers": customers,

            "Budget_Match": budget_label,

            "Remaining_Budget": remaining_budget,

            "Risk_Level": risk_level,

            "Reasons": reasons,

        })


    # --------------------------------------------------------
    # Sort recommendations
    # --------------------------------------------------------

    recommendations.sort(
        key=lambda x: x["Final_Score"],
        reverse=True
    )


    # --------------------------------------------------------
    # Best recommendation
    # --------------------------------------------------------

    best = recommendations[0]


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "status": "success",

        "location": {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "area": request.area,
        },

        "district": {
            "name": district["district"],
            "population_2023": int(
                population
            ),
            "population_density": round(
                population_density,
                2
            ),
            "urban_proportion": round(
                urban_proportion,
                2
            ),
            "area_sq_km": float(
                district["area_sq_km"]
            ),
            "avg_household_size": float(
                district["avg_household_size"]
            ),
            "annual_growth_rate": round(
                annual_growth_rate,
                2
            ),
            "demand_score": demand_score,
        },

        "budget": request.budget,

        "best_recommendation": best,

        "top_3": recommendations[:3],

        "all_recommendations": recommendations,

    }