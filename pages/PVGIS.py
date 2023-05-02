import pandas as pd

# Define conversion function
def convert_timestamp(timestamp_str):
    timestamp = pd.Timestamp(timestamp_str)
    new_timestamp = timestamp.tz_convert('Europe/Amsterdam').strftime('%m/%d/%Y %H:%M')
    new_index = pd.Timestamp(new_timestamp).tz_localize('Europe/Amsterdam', ambiguous='NaT').tz_convert('UTC')
    new_index_numeric = new_index.value // 10**9 // 900 + 34748
    return new_timestamp



data1 = pd.read_csv('data_original.csv')
data2 = pd.read_csv('52.329895_6.112541_Solcast_PT15M.csv')

# Convert the 'start_time' and 'end_time' columns to a datetime format
data2['PeriodStart'] = pd.to_datetime(data2['PeriodStart'])
data2.drop(['PeriodEnd'], axis=1,inplace=True)
data2.drop(data2.index[:92], inplace=True)
data2['PeriodStart'] = data2['PeriodStart'].apply(lambda x: convert_timestamp(x))
data2.drop(data2.index[-486+96:], inplace=True)



# Determine the column lengths of each dataframe
len_data1 = len(data1.index)
len_data2 = len(data2.index)

# Drop the first 95 rows and the last n rows of data2, where n is the difference in row lengths between data2 and data1


# Print the resulting dataframes
print(data1.head(2))
print(data2.head(2))
print('-------------------')
print(data1.tail(2))
print(data2.tail(2))
