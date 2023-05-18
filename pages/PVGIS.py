import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import datetime as dt
# Define conversion function
def convert_timestamp(timestamp_str):
    timestamp = pd.Timestamp(timestamp_str)
    new_timestamp = timestamp.tz_convert('Europe/Amsterdam').strftime('%m/%d/%Y %H:%M')
    
        
    new_index = pd.Timestamp(new_timestamp).tz_localize('Europe/Amsterdam', ambiguous='NaT').tz_convert('UTC')
    new_index_numeric = new_index.value // 10**9 // 900 + 34748
    return new_timestamp

def prepare_data():
    data1 = pd.read_csv(r'data_original.csv')
    data2 = pd.read_csv(r'52.329895_6.112541_Solcast_PT15M.csv')
    data1.set_index(pd.DatetimeIndex(data1['Time']), inplace=True)
    data1 = data1.resample('30T').mean()
    data1.dropna()
    # Convert the 'start_time' and 'end_time' columns to a datetime format
    data2['PeriodStart'] = pd.to_datetime(data2['PeriodStart'])
    data2.drop(['PeriodEnd'], axis=1,inplace=True)
    data2.drop(data2.index[:92], inplace=True)
    data2['PeriodStart'] = data2['PeriodStart'].apply(lambda x: convert_timestamp(x))
    data2.drop(data2.index[-486+96:], inplace=True)
    data2.set_index(pd.DatetimeIndex(data2['PeriodStart']), inplace=True)
    data2.drop(['PeriodStart'], axis=1,inplace=True)
    df = pd.merge(data1, data2, how='outer', left_index=True, right_index=True)
    df.drop(['General Demand (W)','EV Demand (W)','Heating Demand (W)'], axis=1,inplace=True)
    
    return df
# Drop the first 95 rows and the last n rows of data2, where n is the difference in row lengths between data2 and data1

def prepare_data():
    # Prepare the data as per your requirements
    df = ...
    return df

def preprocess_data(df):
    df['Timestampt'] = df.index
    df['TimestamptMonth'] = df['Timestampt'].dt.month
    df['TimestamptDay'] = df['Timestampt'].dt.day
    df['TimestamptHour'] = df['Timestampt'].dt.hour
    df['TimestamptMinute'] = df['Timestampt'].dt.minute
    df = df.dropna(subset=['PV (W)'])
    return df

def train_model(df):
    X_train = df[['TimestamptMinute', 'TimestamptHour', 'TimestamptMonth', 'TimestamptDay', 'CloudOpacity', 'Dhi', 'Dni', 'Ebh', 'Ghi']]
    y_train = df['PV (W)']
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    model = joblib.load(filename)
    return model

def preprocess_new_data(new_data):
    new_data.index = pd.to_datetime(new_data['period_end']).dt.tz_convert(None)
    new_data.drop(['period_end','period'], axis=1, inplace=True)
    new_data['Timestampt'] = new_data.index
    new_data['TimestamptMonth'] = new_data['Timestampt'].dt.month
    new_data['TimestamptDay'] = new_data['Timestampt'].dt.day
    new_data['TimestamptHour'] = new_data['Timestampt'].dt.hour
    new_data['TimestamptMinute'] = new_data['Timestampt'].dt.minute
    return new_data

def predict_new_data(model, new_data):
    X_new = new_data[['TimestamptMinute', 'TimestamptHour', 'TimestamptMonth', 'TimestamptDay', 'CloudOpacity', 'Dhi', 'Dni', 'Ebh', 'Ghi']]
    y_pred = model.predict(X_new)
    y_pred_df = pd.DataFrame(y_pred, columns=["PV_prediction (W)"], index=new_data.index)
    results = pd.concat([new_data, y_pred_df], axis=1)
    return results

def plot_results(df_filtered, results):
    fig, ax = plt.subplots()
    dt_range = pd.date_range(start='2021-04-26 11:00:00', end='2021-05-03 11:00:00', freq='30T')
    x = np.arange(len(dt_range))
    ax.plot(x, results['PV_prediction (W)'], color='red', label='Predicted PV')
    ax.plot(x, df_filtered['PV (W)'], color='blue', label='Original PV')
    ax.set_title('PV Prediction vs. Original PV')
    ax.legend()
    return fig

def main():
    df = prepare_data()
    new_data = pd.read_csv('estimated_actuals.csv')
    new_data.rename(columns={'ghi':'Ghi', 'ebh':'Ebh', 'dni':'Dni', 'dhi':'Dhi', 'cloud_opacity':'CloudOpacity'}, inplace=True)
    new_data['PV (W)'] = np.zeros(len(new_data))
    
    df = preprocess_data(df)
    
    # Split data into training and validation sets
    X_train = df[['TimestamptMinute', 'TimestamptHour', 'TimestamptMonth', 'TimestamptDay', 'CloudOpacity', 'Dhi', 'Dni', 'Ebh', 'Ghi']]
    y_train = df['PV (W)']
    X_val = df[['TimestamptMinute', 'TimestamptHour', 'TimestamptMonth', 'TimestamptDay', 'CloudOpacity', 'Dhi', 'Dni', 'Ebh', 'Ghi']]
    y_val = df['PV (W)']

    # Create and fit linear regression model
    model = train_model(df)

    # Save trained model to file
    save_model(model, 'trained_model.joblib')

    # Preprocess new data
    new_data = preprocess_new_data(new_data)

    # Load saved model
    model = load_model('trained_model.joblib')

    # Make predictions on new data
    results = predict_new_data(model, new_data)

    # Filter original data
    df_filtered = df.loc['2021-04-26 11:00:00':'2021-05-03 11:00:00']

    # Plot results
    fig = plot_results(df_filtered, results)

    # Display the figure in Streamlit
    st.pyplot(fig)

if __name__ == '__main__':
    main()


