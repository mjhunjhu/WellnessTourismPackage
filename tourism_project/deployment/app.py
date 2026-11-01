from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# 1. Locate the trained model inside the repository
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "deployment" / "best_model.joblib"

# These columns must match the features used in train.py.
FEATURE_COLUMNS = [
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


st.set_page_config(
    page_title="Tourism Package Purchase Prediction",
    page_icon="🧳",
    layout="centered",
)

st.title("Tourism Package Purchase Prediction")
st.write(
    "Enter the customer details below to predict whether the customer "
    "is likely to purchase the tourism package."
)


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error(
        f"Trained model not found: {MODEL_PATH}. Make sure "
        "deployment/best_model.joblib is committed to the repository."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"The trained model could not be loaded: {exc}")
    st.stop()


with st.form("customer_prediction_form"):
    st.subheader("Customer information")
    left, right = st.columns(2)

    with left:
        age = st.number_input("Age", min_value=18, max_value=100, value=36)
        contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Free Lancer", "Small Business", "Large Business"],
        )
        gender = st.selectbox("Gender", ["Female", "Male", "Fe Male"])
        marital_status = st.selectbox(
            "Marital Status", ["Single", "Divorced", "Married", "Unmarried"]
        )
        monthly_income = st.number_input(
            "Monthly Income", min_value=0.0, max_value=200000.0,
            value=22418.0, step=500.0,
        )
        number_of_trips = st.number_input(
            "Number of Trips", min_value=0.0, max_value=50.0,
            value=3.0, step=1.0,
        )
        passport = st.selectbox(
            "Has Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No"
        )

    with right:
        pitch_duration = st.number_input(
            "Duration of Pitch (minutes)", min_value=1.0, max_value=200.0,
            value=14.0, step=1.0,
        )
        persons_visiting = st.number_input(
            "Number of Persons Visiting", min_value=1, max_value=10, value=3
        )
        followups = st.number_input(
            "Number of Follow-ups", min_value=0.0, max_value=20.0,
            value=4.0, step=1.0,
        )
        product_pitched = st.selectbox(
            "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
        )
        property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
        pitch_satisfaction = st.selectbox(
            "Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=2
        )
        own_car = st.selectbox(
            "Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x else "No"
        )
        children_visiting = st.number_input(
            "Number of Children Visiting", min_value=0.0, max_value=10.0,
            value=1.0, step=1.0,
        )
        designation = st.selectbox(
            "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )

    submitted = st.form_submit_button("Predict purchase likelihood")


if submitted:
    input_data = {
        "Age": float(age),
        "TypeofContact": contact,
        "CityTier": int(city_tier),
        "DurationOfPitch": float(pitch_duration),
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": int(persons_visiting),
        "NumberOfFollowups": float(followups),
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": float(property_star),
        "MaritalStatus": marital_status,
        "NumberOfTrips": float(number_of_trips),
        "Passport": int(passport),
        "PitchSatisfactionScore": int(pitch_satisfaction),
        "OwnCar": int(own_car),
        "NumberOfChildrenVisiting": float(children_visiting),
        "Designation": designation,
        "MonthlyIncome": float(monthly_income),
    }
    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

    try:
        prediction = int(model.predict(input_df)[0])
        st.subheader("Prediction")

        if prediction == 1:
            st.success("The model predicts that this customer is likely to purchase the package.")
        else:
            st.info("The model predicts that this customer is unlikely to purchase the package.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            model_classes = list(model.classes_)
            if 1 in model_classes:
                purchase_probability = float(probabilities[model_classes.index(1)])
                st.metric("Estimated purchase probability", f"{purchase_probability:.1%}")

        with st.expander("View submitted input data"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.caption(
            "Confirm that the deployed model was trained with the same feature "
            "columns and preprocessing pipeline used by this app."
        )
