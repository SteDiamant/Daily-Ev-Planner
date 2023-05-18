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
    data2 = pd.read_csv(r'52.329895_6.112541_Solcast_PT15M')
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

df = prepare_data()
new_data = pd.read_csv('../estimated_actuals.csv')
new_data.rename(columns={'ghi':'Ghi','ebh':'Ebh','dni':'Dni','dhi':'Dhi','cloud_opacity':'CloudOpacity'}, inplace=True)
new_data['PV (W)'] = np.zeros(len(new_data))
new_data.index = pd.to_datetime(new_data['period_end'], utc=True).dt.tz_convert(None)
new_data.drop(['period_end','period'], axis=1,inplace=True)
new_data['Timestampt'] = new_data.index
new_data['TimestamptMonth'] = new_data['Timestampt'].dt.month
new_data['TimestamptDay'] = new_data['Timestampt'].dt.day
new_data['TimestamptHour'] = new_data['Timestampt'].dt.hour
new_data['TimestamptMinute'] = new_data['Timestampt'].dt.minute
df['Timestampt'] = df.index
df['TimestamptMonth'] = df['Timestampt'].dt.month
df['TimestamptDay'] = df['Timestampt'].dt.day
df['TimestamptHour'] = df['Timestampt'].dt.hour
df['TimestamptMinute'] = df['Timestampt'].dt.minute
df = df.dropna(subset=['PV (W)'])
st.dataframe(df)
st.dataframe(new_data)
# # Split data into training and validation sets

X_train = df[['TimestamptMinute','TimestamptHour','TimestamptMonth','TimestamptDay','CloudOpacity', 'Dhi', 'Dni', 'Ebh', 'Ghi']]
y_train = df['PV (W)']
X_val = df[['TimestamptMinute','TimestamptHour','TimestamptMonth','TimestamptDay','CloudOpacity', 'Dhi', 'Dni', 'Ebh', 'Ghi']]
y_val = df['PV (W)']

# # Create and fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# # Make predictions on validation set and calculate mean squared error
y_val_pred = model.predict(X_val)



# # Save trained model to file
joblib.dump(model, 'trained_model.joblib')



# Preprocess new data (same preprocessing steps as training data)

# Load saved model
model = joblib.load('trained_model.joblib')

# Make predictions on new data
X_new = new_data[['TimestamptMinute','TimestamptHour','TimestamptMonth','TimestamptDay','CloudOpacity', 'Dhi', 'Dni', 'Ebh', 'Ghi']]
y_pred = model.predict(X_new)
y_pred_df = pd.DataFrame(y_pred, columns=["PV_prediction (W)"], index=new_data.index)

# Combine original and predicted dataframes horizontally
results = pd.concat([new_data, y_pred_df],axis=1)
# Filter original data from 26/04 to 05/03
df_filtered = df.loc['2021-04-26 11:00:00':'2021-05-03 11:00:00']



# create a new figure and axes
fig, ax = plt.subplots()
dt_range = pd.date_range(start='2021-04-26 11:00:00', end='2021-05-03 11:00:00', freq='30T')
x = np.arange(len(dt_range))
# plot the predicted PV line on the axes
ax.plot(x, results['PV_prediction (W)'], color='red', label='Predicted PV')

# plot the original PV line on the axes
ax.plot(x, df_filtered['PV (W)'], color='blue', label='Original PV')

# set the title and legend
ax.set_title('PV Prediction vs. Original PV')
ax.legend()

# save the figure


# display the figure in Streamlit
st.pyplot(fig)


