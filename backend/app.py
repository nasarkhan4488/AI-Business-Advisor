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
    version="5.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "business_model.pkl"
)

BUSINESS_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "business_data.pkl"
)

CENSUS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "population",
    "kp_district_population_2023.csv"
)


# ============================================================
# LOAD MODEL + DATA
# ============================================================

print("=" * 60)
print("Loading AI Business Advisor...")
print("=" * 60)

try:
    model = joblib.load(MODEL_PATH)

    business_df = joblib.load(
        BUSINESS_DATA_PATH
    )

    census_df = pd.read_csv(
        CENSUS_PATH
    )

except Exception as e:
    print("ERROR LOADING DATA:")
    print(e)
    raise


# ============================================================
# CLEAN DATA
# ============================================================

business_df.columns = (
    business_df.columns
    .str.strip()
)

census_df.columns = (
    census_df.columns
    .str.strip()
)


# Normalize district names

census_df["district"] = (
    census_df["district"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# READ FEATURES DIRECTLY FROM TRAINED MODEL
# ============================================================

MODEL_FEATURES = list(
    model.feature_names_in_
)

BUSINESS_TYPE_FEATURES = [
    feature
    for feature in MODEL_FEATURES
    if feature.startswith("Business_Type_")
]


# ============================================================
# BUSINESS TYPES
# ============================================================

BUSINESS_TYPES = sorted(
    business_df["Business_Type"]
    .astype(str)
    .unique()
    .tolist()
)


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
# HELPERS
# ============================================================

def get_district_data(
    district_name: str
):

    district_name = (
        district_name
        .strip()
        .upper()
    )

    result = census_df[
        census_df["district"]
        == district_name
    ]

    if result.empty:
        return None

    return result.iloc[0]


def normalize(
    value,
    minimum,
    maximum
):

    if maximum == minimum:
        return 50.0

    score = (
        (
            value - minimum
        )
        /
        (
            maximum - minimum
        )
    ) * 100

    return max(
        0,
        min(
            100,
            score
        )
    )


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_model_input(
    business_data,
    population,
    population_density,
    urban_proportion,
    annual_growth_rate
):

    df = business_data.copy()

    df["Population"] = population

    df["Population_Density"] = (
        population_density
    )

    df["Urban_Proportion"] = (
        urban_proportion
    )

    df["Annual_Growth_Rate"] = (
        annual_growth_rate
    )

    # One-hot encode business type

    encoded = pd.get_dummies(
        df,
        columns=["Business_Type"],
        prefix="Business_Type"
    )

    # Add every feature expected by model

    for feature in MODEL_FEATURES:

        if feature not in encoded.columns:

            encoded[feature] = 0

    # Correct feature order

    encoded = encoded[
        MODEL_FEATURES
    ]

    return encoded


# ============================================================
# DISTRICT DEMAND
# ============================================================

def calculate_district_demand(
    district
):

    population_score = normalize(

        district["population_2023"],

        census_df[
            "population_2023"
        ].min(),

        census_df[
            "population_2023"
        ].max()

    )

    density_score = normalize(

        district["population_density"],

        census_df[
            "population_density"
        ].min(),

        census_df[
            "population_density"
        ].max()

    )

    urban_score = normalize(

        district["urban_proportion"],

        census_df[
            "urban_proportion"
        ].min(),

        census_df[
            "urban_proportion"
        ].max()

    )

    growth_score = normalize(

        district["annual_growth_rate"],

        census_df[
            "annual_growth_rate"
        ].min(),

        census_df[
            "annual_growth_rate"
        ].max()

    )

    demand_score = (

        population_score * 0.40

        + density_score * 0.25

        + urban_score * 0.20

        + growth_score * 0.15

    )

    return round(
        demand_score,
        2
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "message":
            "AI Business Advisor API is running",

        "version":
            "5.0",

        "districts_connected":
            len(census_df),

        "business_types":
            BUSINESS_TYPES,

        "business_type_count":
            len(BUSINESS_TYPES),

        "model_features":
            MODEL_FEATURES,

        "model_business_type_features":
            BUSINESS_TYPE_FEATURES,

    }


# ============================================================
# BUSINESSES
# ============================================================

@app.get("/businesses")
def get_businesses():

    return {

        "count":
            len(BUSINESS_TYPES),

        "business_types":
            BUSINESS_TYPES

    }


# ============================================================
# AREAS
# ============================================================

@app.get("/areas")
def get_areas():

    return {

        "areas":
            sorted(
                business_df[
                    "Area"
                ]
                .astype(str)
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

        "count":
            len(records),

        "districts":
            records

    }


# ============================================================
# SINGLE DISTRICT
# ============================================================

@app.get(
    "/district/{district_name}"
)
def get_district(
    district_name: str
):

    district = get_district_data(
        district_name
    )

    if district is None:

        raise HTTPException(
            status_code=404,
            detail="District not found"
        )

    return {

        "district":
            district["district"],

        "area_sq_km":
            float(
                district["area_sq_km"]
            ),

        "population_2023":
            int(
                district["population_2023"]
            ),

        "population_density":
            float(
                district["population_density"]
            ),

        "urban_proportion":
            float(
                district["urban_proportion"]
            ),

        "avg_household_size":
            float(
                district["avg_household_size"]
            ),

        "annual_growth_rate":
            float(
                district["annual_growth_rate"]
            ),

        "demand_score":
            calculate_district_demand(
                district
            ),

    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict_business(
    request: BusinessRequest
):

    # --------------------------------------------------------
    # Budget validation
    # --------------------------------------------------------

    if request.budget <= 0:

        raise HTTPException(
            status_code=400,
            detail="Budget must be greater than 0"
        )


    # --------------------------------------------------------
    # District
    # --------------------------------------------------------

    district = get_district_data(
        request.district
    )

    if district is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"District "
                f"'{request.district}' "
                f"not found"
            )
        )


    # --------------------------------------------------------
    # District values
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

    demand_score = (
        calculate_district_demand(
            district
        )
    )


    # --------------------------------------------------------
    # Area validation
    # --------------------------------------------------------

    requested_area = (
        request.area.strip()
    )

    area_data = business_df[
        business_df["Area"]
        .astype(str)
        .str.lower()
        ==
        requested_area.lower()
    ].copy()

    if area_data.empty:

        # If requested area doesn't exist,
        # evaluate all available areas.

        area_data = business_df.copy()


    # ========================================================
    # MODEL INPUT
    # ========================================================

    model_input = create_model_input(

        area_data,

        population,

        population_density,

        urban_proportion,

        annual_growth_rate

    )


    # ========================================================
    # ML PREDICTION
    # ========================================================

    probabilities = model.predict_proba(
        model_input
    )


    # Find probability for class 1

    if 1 in model.classes_:

        positive_class_index = list(
            model.classes_
        ).index(1)

        suitable_probability = (
            probabilities[
                :,
                positive_class_index
            ] * 100
        )

    else:

        suitable_probability = (
            model.predict(
                model_input
            ) * 100
        )


    # ========================================================
    # SCORE LIMITS
    # ========================================================

    max_competition = max(

        float(
            business_df[
                "Competition"
            ].max()
        ),

        1

    )

    max_rent = max(

        float(
            business_df[
                "Monthly_Rent"
            ].max()
        ),

        1

    )

    max_customers = max(

        float(
            business_df[
                "Customers"
            ].max()
        ),

        1

    )


    minimum_required_budget = float(

        business_df[
            "Budget_Required"
        ].min()

    )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    recommendations = []


    for index, (
        _,
        business
    ) in enumerate(
        area_data.iterrows()
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
        # Competition score
        # ----------------------------------------------------

        competition_score = (

            1
            -
            competition
            /
            max_competition

        ) * 100


        # ----------------------------------------------------
        # Rent score
        # ----------------------------------------------------

        rent_score = (

            1
            -
            monthly_rent
            /
            max_rent

        ) * 100


        # ----------------------------------------------------
        # Customer score
        # ----------------------------------------------------

        customer_score = (

            customers
            /
            max_customers

        ) * 100


        # ----------------------------------------------------
        # Opportunity score
        # ----------------------------------------------------

        opportunity_score = (

            demand_score * 0.40

            + competition_score * 0.20

            + rent_score * 0.10

            + customer_score * 0.20

            + (
                float(
                    suitable_probability[
                        index
                    ]
                ) * 0.10
            )

        )


        opportunity_score = round(

            max(
                0,
                min(
                    100,
                    opportunity_score
                )
            ),

            2

        )


        # ====================================================
        # BUDGET
        # ====================================================

        budget_ratio = (

            request.budget
            /
            required_budget

        )

        remaining_budget = (

            request.budget
            -
            required_budget

        )


        if budget_ratio >= 1.5:

            budget_match = 100

            budget_label = "Excellent"

            budget_status = (
                "Fully affordable"
            )

        elif budget_ratio >= 1.0:

            budget_match = 85

            budget_label = "Good"

            budget_status = (
                "Affordable"
            )

        elif budget_ratio >= 0.75:

            budget_match = 45

            budget_label = "Near Budget"

            budget_status = (
                "Slightly below estimated "
                "requirement"
            )

        elif budget_ratio >= 0.50:

            budget_match = 20

            budget_label = "Low Budget"

            budget_status = (
                "Additional investment "
                "may be required"
            )

        else:

            budget_match = 5

            budget_label = "Very Low Budget"

            budget_status = (
                "Significant additional "
                "investment may be required"
            )


        # ====================================================
        # FINAL SCORE
        # ====================================================

        final_score = (

            float(
                suitable_probability[
                    index
                ]
            ) * 0.45

            + opportunity_score * 0.35

            + budget_match * 0.20

        )

        final_score = round(

            max(
                0,
                min(
                    100,
                    final_score
                )
            ),

            2

        )


        # ====================================================
        # COMPETITION LEVEL
        # ====================================================

        competition_ratio = (

            competition
            /
            max_competition

        )


        if competition_ratio <= 0.33:

            competition_level = "Low"

        elif competition_ratio <= 0.66:

            competition_level = "Medium"

        else:

            competition_level = "High"


        # ====================================================
        # RISK
        # ====================================================

        if (
            final_score >= 75
            and competition_ratio <= 0.40
        ):

            risk_level = "Low"

        elif (
            final_score >= 55
            and competition_ratio <= 0.70
        ):

            risk_level = "Medium"

        else:

            risk_level = "High"


        # ====================================================
        # SUITABILITY
        # ====================================================

        probability = float(
            suitable_probability[
                index
            ]
        )

        if probability >= 75:

            suitability = "High"

        elif probability >= 50:

            suitability = "Medium"

        else:

            suitability = "Low"


        # ====================================================
        # REASONS
        # ====================================================

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
                "Low competition"
            )

        elif competition_level == "Medium":

            reasons.append(
                "Manageable competition"
            )


        if budget_label == "Excellent":

            reasons.append(
                "Strong budget compatibility"
            )

        elif budget_label == "Good":

            reasons.append(
                "Fits your budget"
            )

        elif budget_label == "Near Budget":

            reasons.append(
                "Close to your available budget"
            )


        if customers >= business_df[
            "Customers"
        ].median():

            reasons.append(
                "Good customer potential"
            )


        if request.budget < required_budget:

            shortfall = (

                required_budget
                -
                request.budget

            )

            reasons.append(

                "Additional budget needed: "
                f"PKR {shortfall:,.0f}"

            )


        if not reasons:

            reasons.append(
                "Based on available business "
                "and district data"
            )


        # ====================================================
        # ADD RESULT
        # ====================================================

        recommendations.append({

            "Business_Type":
                str(
                    business[
                        "Business_Type"
                    ]
                ),

            "Area":
                str(
                    business["Area"]
                ),

            "Budget_Required":
                required_budget,

            "Suitable_Probability":
                round(
                    probability,
                    2
                ),

            "Suitability":
                suitability,

            "Opportunity_Score":
                opportunity_score,

            "Final_Score":
                final_score,

            "Competition":
                competition,

            "Competition_Level":
                competition_level,

            "Monthly_Rent":
                monthly_rent,

            "Customers":
                customers,

            "Budget_Match":
                budget_label,

            "Budget_Status":
                budget_status,

            "Budget_Ratio":
                round(
                    budget_ratio,
                    2
                ),

            "Remaining_Budget":
                remaining_budget,

            "Additional_Budget_Required":
                max(
                    0,
                    required_budget
                    -
                    request.budget
                ),

            "Risk_Level":
                risk_level,

            "Reasons":
                reasons

        })


    # ========================================================
    # SORT
    # ========================================================

    recommendations.sort(

        key=lambda item:
            item["Final_Score"],

        reverse=True

    )


    if not recommendations:

        raise HTTPException(

            status_code=500,

            detail=(
                "No recommendations "
                "could be generated."
            )

        )


    # ========================================================
    # TOP RESULTS
    # ========================================================

    best = recommendations[0]

    top_3 = recommendations[:3]


    # ========================================================
    # BUDGET OVERVIEW
    # ========================================================

    if request.budget >= minimum_required_budget:

        budget_overview = (
            "Your budget can cover at least "
            "one available business option."
        )

    else:

        budget_overview = (
            "Your budget is below the minimum "
            "estimated requirement. The AI has "
            "ranked the closest opportunities "
            "and shows the additional budget needed."
        )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "status":
            "success",

        "location": {

            "latitude":
                request.latitude,

            "longitude":
                request.longitude,

            "area":
                request.area

        },

        "district": {

            "name":
                district["district"],

            "population_2023":
                int(population),

            "population_density":
                round(
                    population_density,
                    2
                ),

            "urban_proportion":
                round(
                    urban_proportion,
                    2
                ),

            "area_sq_km":
                float(
                    district[
                        "area_sq_km"
                    ]
                ),

            "avg_household_size":
                float(
                    district[
                        "avg_household_size"
                    ]
                ),

            "annual_growth_rate":
                round(
                    annual_growth_rate,
                    2
                ),

            "demand_score":
                demand_score

        },

        "budget":
            request.budget,

        "budget_overview":
            budget_overview,

        "minimum_required_budget":
            minimum_required_budget,

        "business_type_count":
            len(BUSINESS_TYPES),

        "best_recommendation":
            best,

        "top_3":
            top_3,

        "all_recommendations":
            recommendations

    }


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port
    )