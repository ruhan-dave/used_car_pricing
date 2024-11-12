
import streamlit as st 
from datetime import datetime
import pandas as pd 
import numpy as np
# import scipy as sp
import pickle
# import datetime as dt
import json
import xgboost as xgb
from st_files_connection import FilesConnection
from io import BytesIO
import boto3
# from io import StringIO

# title of the Web App
st.title("How much $ is your car NOW?")
st.write("This application predicts the price of your vehicle in just a minute, with minimal information provided. Note that results are only estimations and real-world numbers differ case-by-case.")

# processor_path = '/Users/owner/Desktop/data_science/ml_course/aipi510/assignment9/preprocessor.pkl'
# model_path = "/Users/owner/Desktop/data_science/ml_course/aipi510/assignment9/xgbr_model.pkl"

bucket_name = 'churn-challenge'

conn = st.connection('s3', type=FilesConnection)
df = conn.read("churn-challenge/x_train.csv", ttl=600)

s3 = boto3.resource('s3')

with BytesIO() as file:
   s3.Bucket("churn-challenge").download_fileobj("preprocessor.pkl", file)
   file.seek(0)    # move back to the beginning after writing
   processor = pickle.load(file)

with BytesIO() as mod:
   s3.Bucket("churn-challenge").download_fileobj("xgbr_model.pkl", mod)
   mod.seek(0)    # move back to the beginning after writing
   model = pickle.load(mod)

# transform the user_input as we have been transforming the data as before
import streamlit as st
import pandas as pd

def user_inputs(df):

    brand = st.selectbox("Select Brand", df["brand"].unique())
    model = st.selectbox("Select Model", df[df['brand']==brand]["model"].unique())
    model_year = st.slider("Model Year", min_value=2000, max_value=2024, step=1)
    milage = st.number_input("Mileage", min_value=0, step=1000)
    fuel_type = st.selectbox("Select Fuel Type", df["fuel_type"].unique())
    engine = st.selectbox("Select Engine", df[(df['fuel_type']==fuel_type) & (df['model']==model)]["engine"].unique())  
    transmission = st.selectbox("Select Transmission", df[df['engine']==engine]["transmission"].unique())
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

    df = pd.DataFrame(data, index=[0])
    return df

def data_transform(df, processor):
    return processor.transform(df)

# Predict with the model 
def predict(model, transformed):
    output = np.rint(model.predict(transformed))
    return output

def main():
    x_input = user_inputs(df)

    # design user interface
    if st.button("Find out"):
        transformed = data_transform(x_input, processor)
        prediction = predict(model, transformed)[0]
        st.markdown(
            f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: green;'>Predicted Price ${round(prediction/1000)}k</div>",
            unsafe_allow_html=True
        )
if __name__ == "__main__":
    main()
