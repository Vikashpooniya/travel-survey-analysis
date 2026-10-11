
import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load saved model and encoders
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/travel_survey_model.pkl")
    encoders = joblib.load("models/label_encoders.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, encoders, feature_columns

model, label_encoders, feature_columns = load_artifacts()

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Travel Experience Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Travel Experience Predictor")
st.write(
    "Enter passenger details and service ratings to predict "
    "their overall travel experience."
)

st.info(
    "This app uses a Random Forest model trained on the "
    "Travel Survey dataset."
)

# -------------------------------
# Passenger and travel details
# -------------------------------
st.header("1. Passenger and Travel Details")

c1, c2 = st.columns(2)

with c1:
    gender = st.selectbox(
        "Gender",
        label_encoders["Gender"].classes_.tolist()
    )

    customer_type = st.selectbox(
        "Customer Type",
        label_encoders["Customer_Type"].classes_.tolist()
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

with c2:
    type_travel = st.selectbox(
        "Type of Travel",
        label_encoders["Type_Travel"].classes_.tolist()
    )

    travel_class = st.selectbox(
        "Travel Class",
        label_encoders["Travel_Class"].classes_.tolist()
    )

    seat_class = st.selectbox(
        "Seat Class",
        label_encoders["Seat_Class"].classes_.tolist()
    )

travel_distance = st.number_input(
    "Travel Distance",
    min_value=0,
    max_value=10000,
    value=1000
)

# -------------------------------
# Service ratings
# -------------------------------
st.header("2. Service Ratings")
st.caption(
    "Choose ratings using the same scale as the original dataset."
)

rating_cols = [
    "Seat_Comfort",
    "Arrival_Time_Convenient",
    "Catering",
    "Platform_Location",
    "Onboard_Wifi_Service",
    "Onboard_Entertainment",
    "Online_Support",
    "Ease_of_Online_Booking",
    "Onboard_Service",
    "Legroom",
    "Baggage_Handling",
    "CheckIn_Service",
    "Cleanliness",
    "Online_Boarding"
]

rating_labels = {
    "Seat_Comfort": "Seat Comfort",
    "Arrival_Time_Convenient": "Arrival Time Convenience",
    "Catering": "Catering",
    "Platform_Location": "Platform Location",
    "Onboard_Wifi_Service": "Onboard Wi-Fi",
    "Onboard_Entertainment": "Onboard Entertainment",
    "Online_Support": "Online Support",
    "Ease_of_Online_Booking": "Ease of Online Booking",
    "Onboard_Service": "Onboard Service",
    "Legroom": "Legroom",
    "Baggage_Handling": "Baggage Handling",
    "CheckIn_Service": "Check-in Service",
    "Cleanliness": "Cleanliness",
    "Online_Boarding": "Online Boarding"
}

ratings = {}

rating_columns = st.columns(2)

for i, col in enumerate(rating_cols):
    with rating_columns[i % 2]:
        ratings[col] = st.selectbox(
            rating_labels[col],
            options=[0, 1, 2, 3, 4, 5],
            index=3,
            key=col
        )

# -------------------------------
# Flight delays
# -------------------------------
st.header("3. Flight Delay Information")

d1, d2 = st.columns(2)

with d1:
    departure_delay = st.number_input(
        "Departure Delay (minutes)",
        min_value=0,
        max_value=10000,
        value=0
    )

with d2:
    arrival_delay = st.number_input(
        "Arrival Delay (minutes)",
        min_value=0,
        max_value=10000,
        value=0
    )

# -------------------------------
# Predict
# -------------------------------
if st.button("Predict Overall Experience", type="primary"):

    # Encode categorical inputs using the saved training encoders
    categorical_values = {
        "Gender": gender,
        "Customer_Type": customer_type,
        "Type_Travel": type_travel,
        "Travel_Class": travel_class,
        "Seat_Class": seat_class
    }

    encoded_values = {}

    for col, value in categorical_values.items():
        encoded_values[col] = int(
            label_encoders[col].transform([value])[0]
        )

    # Create the input row
    input_data = {
        **ratings,
        **encoded_values,
        "Age": age,
        "Travel_Distance": travel_distance,
        "Departure_Delay_in_Mins": departure_delay,
        "Arrival_Delay_in_Mins": arrival_delay
    }

    # Feature engineering: use the same formulas as training
    input_data["Total_Delay"] = (
        departure_delay + arrival_delay
    )

    input_data["Service_Quality_Score"] = sum(
        ratings[col]
        for col in [
            "Seat_Comfort",
            "Catering",
            "Onboard_Wifi_Service",
            "Onboard_Entertainment",
            "Online_Support",
            "Ease_of_Online_Booking",
            "Onboard_Service",
            "Legroom",
            "Baggage_Handling",
            "CheckIn_Service",
            "Cleanliness",
            "Online_Boarding"
        ]
    ) / 12

    # Match pandas.cut bins and category codes used in training
    age_group = pd.cut(
        [age],
        bins=[0, 25, 35, 50, 100],
        labels=["Young", "Adult", "Middle_Age", "Senior"]
    )
    input_data["Age_Group"] = int(age_group.codes[0])

    distance_category = pd.cut(
        [travel_distance],
        bins=[0, 500, 1500, 3000, 10000],
        labels=["Short", "Medium", "Long", "Very_Long"]
    )
    input_data["Distance_Category"] = int(
        distance_category.codes[0]
    )

    # Match exact model feature order
    input_df = pd.DataFrame(
        [[input_data[col] for col in feature_columns]],
        columns=feature_columns
    )

    # Predict
    prediction = model.predict(input_df)[0]

    st.divider()
    st.header("Prediction Result")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        positive_index = classes.index(1) if 1 in classes else None

        if positive_index is not None:
            probability = float(probabilities[positive_index])

            st.metric(
                "Predicted Satisfaction Probability",
                f"{probability:.1%}"
            )

    if prediction == 1:
        st.success("The model predicts a positive overall experience.")
    else:
        st.warning("The model predicts a negative overall experience.")

    st.caption(
        "This is a machine-learning prediction, not a guarantee "
        "of a passenger's actual experience."
    )
