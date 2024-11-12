import streamlit as st
import pandas as pd
import numpy as np
import pickle
import boto3
from io import BytesIO
from st_files_connection import FilesConnection


@st.cache_resource
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets.get("AWS_REGION", "us-west-1")
    )

@st.cache_data
def load_file_from_s3(bucket_name, file_key):
    s3_client = get_s3_client()
    with BytesIO() as file_obj:
        s3_client.download_fileobj(bucket_name, file_key, file_obj)
        file_obj.seek(0)
        return pickle.load(file_obj)

@st.cache_data
def load_csv_from_s3(connection, file_path):
    return connection.read(file_path, ttl=600)

bucket_name = 'churn-challenge'
conn = st.connection('s3', type=FilesConnection)

df = load_csv_from_s3(conn, f"{bucket_name}/x_train.csv")
processor = load_file_from_s3(bucket_name, "preprocessor.pkl")
model = load_file_from_s3(bucket_name, "xgbr_model.pkl")

# User input collection function
def user_inputs(df):
    brand = st.selectbox("Select Brand", df["brand"].unique())
    model = st.selectbox("Select Model", df[df['brand'] == brand]["model"].unique())
    model_year = st.slider("Model Year", min_value=2000, max_value=2024, step=1)
    milage = st.number_input("Mileage", min_value=0, step=1000)
    fuel_type = st.selectbox("Select Fuel Type", df["fuel_type"].unique())
    engine = st.selectbox("Select Engine", df[(df['fuel_type'] == fuel_type) & (df['model'] == model)]["engine"].unique())
    transmission = st.selectbox("Select Transmission", df[df['engine'] == engine]["transmission"].unique())
    accident = st.selectbox("Accident History", ["Yes", "No"])  # Assuming binary options

    data = {
        'brand': brand,
        'model': model,
        'model_year': model_year,
        'milage': milage,
        'fuel_type': fuel_type,
        'engine': engine,
        'transmission': transmission,
        'accident': accident
    }
    return pd.DataFrame(data, index=[0])

# Transform data using the loaded processor
def data_transform(input_df, processor):
    return processor.transform(input_df)

# Predict with the loaded model
def predict_price(model, transformed_data):
    output = np.rint(model.predict(transformed_data))
    return output

# Main application with buttons and sliders
def main():
    st.title("How much $ is your car NOW?")
    st.write("This application predicts the price of your vehicle in just a minute, with minimal information provided. Note that results are only estimations and real-world numbers differ case-by-case.")

    # Collect user inputs
    user_data = user_inputs(df)

    # When the "Find out" button is pressed
    if st.button("Find out"):
        # Transform user data and make prediction
        transformed_data = data_transform(user_data, processor)
        predicted_price = predict_price(model, transformed_data)[0]
        
        # Display the predicted price in a formatted style
        st.markdown(
            f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: green;'>Predicted Price: ${round(predicted_price / 1000)}k</div>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
