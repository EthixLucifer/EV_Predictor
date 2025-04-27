import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def load_model():
    return RandomForestRegressor(n_estimators=100, random_state=42)

# Load your data (replace with your data loading logic)
data = pd.read_csv('.\EV_RangeA1.csv')  # Assuming your data is in a CSV file
def fill_missing_values(data):
    """Fills missing values in the data with the mean."""
    return data.fillna(data.mean(), inplace=False)  # Avoid modifying original data

def prepare_data(data):
    """Prepares the data for prediction."""
    data = fill_missing_values(data.copy())  # Operate on a copy
    selected_columns = ['No_of_wheels', 'Battery_Capacity', 'Vehicle_mass_kg', 'Driver_mass_kg',
                        'C_rr', 'Grade', 'area', 'air_density', 'radius_of_wheel', 'Gear_ratio',
                        'transmission_efficiency', 'motor_efficiency', 'Gross_vehivle_mass', 'g',
                        'motor_controller_efficiency', 'coefficient_of_drag']
    return data[selected_columns]

def split_data(data):
    """Splits data into training and testing sets."""
    X = prepare_data(data.drop('Battery_Capacity_available', axis=1))
    y = data['Battery_Capacity_available']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_train, y_train):
    """Evaluates the model on the training data."""
    y_train_pred = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    return mae_train

def create_ui(model):
    """Creates the Streamlit UI for user interaction and prediction."""

    selected_columns = ['No_of_wheels', 'Battery_Capacity', 'Vehicle_mass_kg', 'Driver_mass_kg',
                        'C_rr', 'Grade', 'area', 'air_density', 'radius_of_wheel', 'Gear_ratio',
                        'transmission_efficiency', 'motor_efficiency', 'Gross_vehivle_mass', 'g',
                        'motor_controller_efficiency', 'coefficient_of_drag','Distance_travel']

    st.title("EV Range Predictor")

    st.subheader("Enter Vehicle Details:")

    user_input = {}
    for column in selected_columns:
        user_input[column] = st.number_input(column, min_value=0.0000, step=0.00001)  # Adjust min/max values as needed

    if st.button("Predict Battery Capacity Remaining"):
        user_input_df = pd.DataFrame([user_input])
        prediction = model.predict(prepare_data(user_input_df))[0]
        st.write(f"Predicted Battery Capacity Available: {prediction:.2f}")

if __name__ == "__main__":
    # Load data and split into training/testing sets (uncomment if needed)
    X_train, X_test, y_train, y_test = split_data(data)
    model = load_model()
    model.fit(X_train, y_train)
    create_ui(model)