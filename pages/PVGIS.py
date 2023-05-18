import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Define conversion function
def convert_timestamp(timestamp_str):
    timestamp = pd.Timestamp(timestamp_str)
    new_timestamp = timestamp.tz_convert('Europe/Amsterdam').strftime('%m/%d/%Y %H:%M')
    
    new_index = pd.Timestamp(new_timestamp).tz_localize('Europe/Amsterdam', ambiguous='NaT').tz_convert('UTC')
    new_index_numeric = new_index.value // 10**9 // 900 + 34748
    return new_timestamp

# Function to prepare data
def prepare_data():
    data1 = pd.read_csv(r'data_original.csv')
    data2 = pd.read_csv(r'52.329895_6.112541_Solcast_PT15M.csv')

    # Data1 preprocessing
    data1['Time'] = pd.to_datetime(data1['Time'])
    data1.set_index('Time', inplace=True)
    data1 = data1.resample('30T').mean().dropna()

    # Data2 preprocessing
    data2['PeriodStart'] = pd.to_datetime(data2['PeriodStart']).dt.tz_convert('Europe/Amsterdam')
    data2.drop(['PeriodEnd'], axis=1, inplace=True)
    data2 = data2.iloc[95:-486+96, :]
    data2.set_index('PeriodStart', inplace=True)

    # Merge data1 and data2
    df = pd.merge(data1, data2, how='outer', left_index=True, right_index=True)
    df.drop(['General Demand (W)', 'EV Demand (W)', 'Heating Demand (W)'], axis=1, inplace=True)
    
    return df

# Function to preprocess new data
def preprocess_new_data(new_data):
    new_data.rename(columns={'ghi':'Ghi','ebh':'Ebh','dni':'Dni','dhi':'Dhi','cloud_opacity':'CloudOpacity'}, inplace=True)
    new_data['PV (W)'] = np.zeros(len(new_data))
    new_data.set_index(pd.to_datetime(new_data['period_end']), inplace=True)
    new_data.drop(['period_end', 'period'], axis=1, inplace=True)
    new_data['Timestampt'] = new_data.index
    new_data['TimestamptMonth'] = new_data['Timestampt'].dt.month
    new_data['TimestamptDay'] = new_data['Timestampt'].dt.day
    new_data['TimestamptHour'] = new_data['Timestampt'].dt.hour
    new_data['TimestamptMinute'] = new_data['Timestampt'].dt.minute
    
    return new_data

# Function to split data into training and validation sets
def split_data(df):
    X = df[['TimestamptMinute', 'TimestamptHour', 'TimestamptMonth', 'TimestamptDay', 'CloudOpacity', 'Dhi', 'Dni', 'Ebh', 'Ghi']]
    y = df['PV (W)']
    return X, y

# Function to train the linear regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to save the trained model
def save_model(model, filename):
    joblib.dump(model, filename)

# Function to load the trained model
def load_model(filename):
    model = joblib.load(filename)
    return model

# Function to make predictions on new data
def make_predictions(model, X_new):
    y_pred = model.predict(X_new)
    return pd.DataFrame(y_pred, columns=["PV_prediction (W)"], index=X_new.index)

#Function to visualize the results
def visualize_results(df_filtered, results):
    fig, ax = plt.subplots()
    x = np.arange(len(df_filtered))
    ax.plot(x, results['PV_prediction (W)'], color='red', label='Predicted PV')
    ax.plot(x, df_filtered['PV (W)'], color='blue', label='Original PV')

    ax.set_title('PV Prediction vs. Original PV')
    ax.legend()

    return fig

#Main code
def main():
    df = prepare_data()
    new_data = pd.read_csv(r'estimated_actuals.csv')
    df_filtered = df.loc['2021-04-26 11:00:00':'2021-05-03 11:00:00']

    new_data = preprocess_new_data(new_data)

    X_train, y_train = split_data(df)
    X_val, y_val = X_train, y_train

    model = train_model(X_train, y_train)
    save_model(model, 'trained_model.joblib')

    model = load_model('trained_model.joblib')
    y_pred_df = make_predictions(model, new_data[['TimestamptMinute', 'TimestamptHour', 'TimestamptMonth', 'TimestamptDay', 'CloudOpacity', 'Dhi', 'Dni', 'Ebh', 'Ghi']])
    results = pd.concat([new_data, y_pred_df], axis=1)

    fig = visualize_results(df_filtered, results)

    st.pyplot(fig)

if __name__ == "__main__":
    main()