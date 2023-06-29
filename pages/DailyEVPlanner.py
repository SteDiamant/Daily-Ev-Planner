import pandas as pd
import numpy as np
from datetime import time,timedelta,datetime
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from html2markdown import convert
import calendar



def split_dataframe_by_day(df):
        days = [df[i:i+96] for i in range(0, len(df), 96)]
        return days


def ETL():
    df = pd.read_csv('data_original.csv')
    df['Time'] = pd.to_datetime(df['Time'])
    df['Month'] = df['Time'].dt.month
    df['DayOfMonth'] = df['Time'].dt.day
    df['DayOfWeek'] = df['Time'].dt.dayofweek
    df['Hour'] = df['Time'].dt.hour
    df['Minute'] = df['Time'].dt.minute
    df.drop(columns=['EV Demand (W)'],inplace=True)
    df['PV (W)']=df['PV (W)']*(327/4)
    df['TotalDemand (W)']=df['General Demand (W)']+df['Heating Demand (W)']
    df['Imbalance (W)']=df['TotalDemand (W)']+df['PV (W)']
    power_imported = np.where(df['Imbalance (W)'] > 0, df['TotalDemand (W)']+df['PV (W)'],0)
    power_wasted = np.where(df['Imbalance (W)'] < 0, df['TotalDemand (W)']+df['PV (W)'],0)
    df['Power Imported (W)'] = power_imported
    df['Power Wasted (W)'] = power_wasted
    days=split_dataframe_by_day(df)
    
    return days

class HtmlGenerator():
        
   def battery_lvl1(id,value):

        # Calculate the percentage of battery charge
        charge_percentage = value / MAXIMUM_CAR_CAPACITY
        
        # Set the height of the #charge div based on the charge percentage
        charge_height = round(charge_percentage * 100, 2)
        if charge_height > 100:
            st.error("The battery capacity cannot be more than 100%")
        elif charge_height < 0:
            st.error("The battery capacity cannot be less than 0%")
        else:
            # Replace the placeholder {value} in the HTML code with the actual value and charge height
            html_code = f"""
            
            <html lang="en">
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1.0" />
                <title>Detect Battery Status</title>
                <!-- Google Fonts -->
                <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet" />
                <!-- Stylesheet -->
                <style>
                    /* CSS code for battery status interface */

                    
                    .container {{
                        width: 300px;
                        margin: 50px auto;
                        text-align: center;
                    }}

                    #battery {{
                        position: relative;
                        margin: 20px auto;
                        width: 80px;
                        height: 120px;
                        border-radius: 10px;
                        border: 5px solid #333;
                    }}

                    #charge{id} {{
                        position: absolute;
                        bottom: 0;
                        width: 100%;
                        
                        background-color: #ff9800;
                        border-radius: 0 0 10px 10px;
                    }}

                    #charge-level {{
                        margin-top: -50px;
                        font-family: 'Roboto Mono', monospace;
                        font-size: 24px;
                        font-weight: 500;
                        color: #333;
                    }}

                    #charging-time {{
                        font-family: 'Roboto Mono', monospace;
                        font-size: 18px;
                        color: #333;
                    }}
                </style>
                </head>
                <body>
                    <div class="container{id}">
                        <div id="charge-level{id}">
                            
                            <h3>Car{id} Battery Level</h3>
                            <div id="battery">
                                <div id="charge-level">{charge_height}%<br>
                                    <div id="charge{id}" style="height:{charge_height}%;">
                                    
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </body>

            </html>
            """
            
            
            markdown_code = convert(html_code)
            return markdown_code


class ProfileGenerator:
    def crete_charge_profile(df,start_time,end_time,start_date,end_date,id,scale_factor,initial_charge) :
        date_range = pd.date_range(start=datetime.combine(start_date, start_time), end=datetime.combine(end_date, end_time), freq='15min')
        charge_profile = pd.DataFrame(index=date_range)
        charge_values=scale_factor*np.random.randint(low=6400, high=7000, size=len(date_range))
        sum_values=charge_values.sum()
        i=1
        while sum_values > MAXIMUM_CAR_CAPACITY-initial_charge:
            charge_values[-i] = 0
            sum_values=charge_values.sum()
            i=i+1
        charge_profile[f'EV{id}_charge (W)'] = pd.Series(data=charge_values, index=charge_profile.index)
        
        charge_profile.index.name = 'Time'
        
        return charge_profile
    
    def crete_discharge_profile(df,start_time,end_time,start_date,end_date,id,scale_factor):
        date_range = pd.date_range(start=datetime.combine(start_date, start_time), end=datetime.combine(end_date, end_time), freq='15min')
        discharge_profile = pd.DataFrame(index=date_range)
        discharge_values=scale_factor*np.random.randint(low=-7000, high=-6400, size=len(date_range))
        sum_values=discharge_values.sum()
        i=1
        while abs(sum_values) > 0.8*(MAXIMUM_CAR_CAPACITY):
            discharge_values[-i] = 0
            sum_values=discharge_values.sum()
            i+=1
        discharge_profile[f'EV{id}_discharge (W)'] = pd.Series(data=discharge_values, index=discharge_profile.index)
        discharge_profile.index.name = 'Time'
        
        return discharge_profile


class MergeProfiles:
    def merge_charge_profile(df,charge_profile):
        merged_df = pd.concat([charge_profile.set_index('Time'), df.set_index('Time')], axis=1)
        merged_df.fillna(0, inplace=True)
        merged_df.drop_duplicates(inplace=True)
        
        merged_df['Total_EV_Charge (W)']=merged_df['EV1_charge (W)']+merged_df['EV2_charge (W)']+merged_df['EV3_charge (W)']+merged_df['EV4_charge (W)']
        merged_df['TotalImbalance']=merged_df['Imbalance (W)']+merged_df['Total_EV_Charge (W)']
        return merged_df
    
    def merge_discharge_profile(df,discharge_profile):
        merged_df = pd.concat([discharge_profile.set_index('Time'), df.set_index('Time')], axis=1)
        merged_df.fillna(0, inplace=True)
        merged_df.drop_duplicates(inplace=True)
        
        merged_df['Total_EV_DisCharge (W)']=merged_df['EV1_discharge (W)']+merged_df['EV2_discharge (W)']+merged_df['EV3_discharge (W)']+merged_df['EV4_discharge (W)']
        merged_df['TotalImbalance']=merged_df['Imbalance (W)']+merged_df['EV1_discharge (W)']+merged_df['EV2_discharge (W)']+merged_df['EV3_discharge (W)']+merged_df['EV4_discharge (W)']
        return merged_df
    

def create_day_charge_profile(day, start_charge_times, end_charge_times,SCALE_FACTORS_CHARGE,initial_charge):
    # Get unique month and day values from the 'day' dataframe
    __get__month = int(day['Month'].unique()[0])  # use index 0 to get the first (and only) element
    __get__day = int(day['DayOfMonth'].unique()[0])

    # Create start and end datetime obkects
    start_date = datetime(year=2021, month=__get__month, day=__get__day)
    end_date = start_date  # set end date equal to start date

    # Create charge profiles for all four EVs
    ev1 = ProfileGenerator.crete_charge_profile(day, start_charge_times[0], end_charge_times[0], start_date, end_date, id=1, scale_factor=SCALE_FACTORS_CHARGE[0],initial_charge=initial_charge[0])
    ev2 = ProfileGenerator.crete_charge_profile(day, start_charge_times[1], end_charge_times[1], start_date, end_date, id=2, scale_factor=SCALE_FACTORS_CHARGE[1],initial_charge=initial_charge[1])
    ev3 = ProfileGenerator.crete_charge_profile(day, start_charge_times[2], end_charge_times[2], start_date, end_date, id=3, scale_factor=SCALE_FACTORS_CHARGE[2],initial_charge=initial_charge[2])
    ev4 = ProfileGenerator.crete_charge_profile(day, start_charge_times[3], end_charge_times[3], start_date, end_date, id=4, scale_factor=SCALE_FACTORS_CHARGE[3],initial_charge=initial_charge[3])

    # Merge the charge profiles into a single dataframe
    charge_data = pd.concat([ev1, ev2, ev3, ev4], axis=1)
    charge_data.reset_index(inplace=True)
    charge_data.fillna(0, inplace=True)

    # Merge the charge profile with the original dataframe
    merged_df = MergeProfiles.merge_charge_profile(day, charge_data)
    
    return merged_df
def create_day_discharge_profile(day, start_charge_times, end_charge_times,SCALE_FACTORS_DISCHARGE):
    # Get unique month and day values from the 'day' dataframe
    __get__month = int(day['Month'].unique()[0])  # use index 0 to get the first (and only) element
    __get__day = int(day['DayOfMonth'].unique()[0])

    # Create start and end datetime obkects
    start_date = datetime(year=2021, month=__get__month, day=__get__day)
    end_date = start_date  # set end date equal to start date

    # Create charge profiles for all four EVs
    ev1 = ProfileGenerator.crete_discharge_profile(day, start_charge_times[0], end_charge_times[0], start_date, end_date, id=1,scale_factor=SCALE_FACTORS_DISCHARGE[0])
    ev2 = ProfileGenerator.crete_discharge_profile(day, start_charge_times[1], end_charge_times[1], start_date, end_date, id=2,scale_factor=SCALE_FACTORS_DISCHARGE[1])
    ev3 = ProfileGenerator.crete_discharge_profile(day, start_charge_times[2], end_charge_times[2], start_date, end_date, id=3,scale_factor=SCALE_FACTORS_DISCHARGE[2])
    ev4 = ProfileGenerator.crete_discharge_profile(day, start_charge_times[3], end_charge_times[3], start_date, end_date, id=4,scale_factor=SCALE_FACTORS_DISCHARGE[3])

    # Merge the charge profiles into a single dataframe
    charge_data = pd.concat([ev1, ev2, ev3, ev4], axis=1)
    charge_data.reset_index(inplace=True)
    charge_data.fillna(0, inplace=True)

    # Merge the charge profile with the original dataframe
    merged_df = MergeProfiles.merge_discharge_profile(day, charge_data)
    
    return merged_df

def calculateTotalEnergy_EV_Charge(df):
    population = 0
    population += df['EV1_charge (W)'].sum()*0.00025
    population += df['EV2_charge (W)'].sum()*0.00025
    population += df['EV3_charge (W)'].sum()*0.00025
    population += df['EV4_charge (W)'].sum()*0.00025
    return population , [df['EV1_charge (W)'].sum()*0.00025,df['EV2_charge (W)'].sum()*0.00025,df['EV3_charge (W)'].sum()*0.00025,df['EV4_charge (W)'].sum()*0.00025]

def calculateTotalEnergy_EV_DisCharge(df):
    population = 0
    population += df['EV1_discharge (W)'].sum()*0.00025
    population += df['EV2_discharge (W)'].sum()*0.00025
    population += df['EV3_discharge (W)'].sum()*0.00025
    population += df['EV4_discharge (W)'].sum()*0.00025
    return population , [df['EV1_discharge (W)'].sum()*0.00025,df['EV2_discharge (W)'].sum()*0.00025,df['EV3_discharge (W)'].sum()*0.00025,df['EV4_discharge (W)'].sum()*0.00025]

def count_positive_charge_negative_imbalance(df):
    count=0
    count1=0
    count  += len(df[(df['EV1_charge (W)'] > 0) & (df['TotalImbalance'] <= 0)])
    count  += len(df[(df['EV2_charge (W)'] > 0) & (df['TotalImbalance'] <= 0)])
    count  += len(df[(df['EV3_charge (W)'] > 0) & (df['TotalImbalance'] <= 0)])
    count  += len(df[(df['EV4_charge (W)'] > 0) & (df['TotalImbalance'] <= 0)])
    count1 += len(df[(df['EV1_charge (W)'] > 0) & (df['TotalImbalance'] >= 0)])
    count1 += len(df[(df['EV2_charge (W)'] > 0) & (df['TotalImbalance'] >= 0)])
    count1 += len(df[(df['EV3_charge (W)'] > 0) & (df['TotalImbalance'] >= 0)])
    count1 += len(df[(df['EV4_charge (W)'] > 0) & (df['TotalImbalance'] >= 0)])
    total_count=len(df['TotalImbalance']>0)
    

    
    return count,count1 ,total_count, [len(df[(df['EV1_charge (W)'] > 0) & (df['TotalImbalance'] < 0)]),
                                       len(df[(df['EV2_charge (W)'] > 0) & (df['TotalImbalance'] < 0)]),
                                       len(df[(df['EV3_charge (W)'] > 0) & (df['TotalImbalance'] < 0)]),
                                       len(df[(df['EV4_charge (W)'] > 0) & (df['TotalImbalance'] < 0)])]
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def save_info(start_charge_time1,end_charge_time1,
                                              start_charge_time2,end_charge_time2,
                                              start_charge_time3,end_charge_time3,
                                              start_charge_time4,end_charge_time4,
                                              start_discharge_time1,end_discharge_time1,
                                              start_discharge_time2,end_discharge_time2,
                                              start_discharge_time3,end_discharge_time3,
                                              start_discharge_time4,end_discharge_time4,
                                              DAY,
                                              SCALE_FACTOR1_CHARGE,SCALE_FUCTOR2_CHARGE,
                                              SCALE_FUCTOR3_CHARGE,SCALE_FUCTOR4_CHARGE,
                                              SCALE_FACTOR1_DISHCARGE,SCALE_FUCTOR2_DISCHARGE,
                                              SCALE_FUCTOR3_DISCHARGE,SCALE_FUCTOR4_DISCHARGE):
    filename = f"EV_Profile{DAY}.txt"
    with open(filename, "w") as file:
        file.write(f"EV1 Charge start Time: {start_charge_time1} end Time {end_charge_time1} ScaleFactor {SCALE_FACTOR1_CHARGE}\n")
        file.write(f"EV2 Charge start Time: {start_charge_time2} end Time {end_charge_time2} ScaleFactor {SCALE_FUCTOR2_CHARGE}\n")
        file.write(f"EV3 Charge start Time: {start_charge_time3} end Time {end_charge_time3} ScaleFactor {SCALE_FUCTOR3_CHARGE}\n")
        file.write(f"EV4 Charge start Time: {start_charge_time4} end Time {end_charge_time4} ScaleFactor {SCALE_FUCTOR4_CHARGE}\n")
        file.write(f"EV1 Discharge start Time: {start_discharge_time1} end Time {end_discharge_time1} ScaleFactor {SCALE_FACTOR1_DISHCARGE}\n")
        file.write(f"EV2 Discharge start Time: {start_discharge_time2} end Time {end_discharge_time2} ScaleFactor {SCALE_FUCTOR2_DISCHARGE}\n")
        file.write(f"EV3 Discharge start Time: {start_discharge_time3} end Time {end_discharge_time3} ScaleFactor {SCALE_FUCTOR3_DISCHARGE}\n")
        file.write(f"EV4 Discharge start Time: {start_discharge_time4} end Time {end_discharge_time4} ScaleFactor {SCALE_FUCTOR4_DISCHARGE}\n")
        file.close()


def plot_pie_chart(labels, values):
    fig, ax = plt.subplots()
    plt.title('Energy Origin Perchentage \n used for charging EVs')
    ax.pie(values, labels=labels, autopct='%1.1f%%',colors=('g','r'))
    
    ax.set_aspect('equal')
    
    return fig

def calculate_energy_storage(df,MAX_CO_CARS,STARTING_CAR_CAPACITY):
    for k in range(1,MAX_CO_CARS+1):
        # initialize an empty DataFrame to store the energy storage values
        df[f'BatteryLVL{k}'] = STARTING_CAR_CAPACITY[k-1]
        
        # set the initial energy storage value
        df.at[df.index[0], f'BatteryLVL{k}'] = STARTING_CAR_CAPACITY[k-1]
        
        # iterate over the rows of the DataFrame
        for i in range(1, len(df)):
            # calculate the energy storage value using the recursive formula
            Ep_t = df.at[df.index[i-1], f'BatteryLVL{k}'] + df.at[df.index[i], f'EV{k}_charge (W)'] + df.at[df.index[i], f'EV{k}_discharge (W)']
            
            # append the new energy storage value to the DataFrame
            df.at[df.index[i], f'BatteryLVL{k}'] = Ep_t
            
        
        # return the energy storage DataFrame
    return df


@st.cache_data
def plot_battery_level(df):
    # create a new figure and axis object
    fig, ax = plt.subplots(1,4, figsize=(8, 5))
    # plot the battery level values
    for i in range(1, 5):
        #df=df.loc[~((df[f'BatteryLVL{i}'] == 0))]
            #ax[i-1].stackplot(df.index, df[f'BatteryLVL{i}'], color='r'if df['TotalImbalance']>0 else 'g')
            ax[i-1].stackplot(df.index, df[f'BatteryLVL{i}'], color='r')

            ax[i-1].axhline(y=MAXIMUM_CAR_CAPACITY, xmin=0, xmax=1, color='r', linestyle='-',label='Maximum Capacity')
            ax[i-1].axhline(y=MAXIMUM_CAR_CAPACITY*0.8, xmin=0, xmax=1, color='r', linestyle='--',label='80% Capacity%')
            ax[i-1].axhline(y=MAXIMUM_CAR_CAPACITY*0.2, xmin=0, xmax=1, color='r', linestyle='--',label='20% Capacity')
            ax[i-1].axhline(y=0, xmin=0, xmax=1, color='r', linestyle='--',label='Minimum Capacity')
            ax[i-1].set_title(f'EV{i}')
            ax[i-1].get_xaxis().set_visible(False)
            ax[i-1].set_xlabel('Time')
    st.pyplot(fig)


def main():
    
    days=ETL()
    day=days[DAY]
    with st.form("my_form"):
        car1,car2,car3,car4=st.columns(4)
        with car1:
            st.title('EV1 :car:')
            ch1,dis1=st.columns(2)
            with ch1:
                with st.expander('EV1 Charge'):
                    start_charge_time1 = st.time_input('start charge1 :clock1: ',value=time(hour=9, minute=30),key='start_charge_time1')
                    end_charge_time1 = st.time_input('end charge1 :checkered_flag:',value=time(hour=15, minute=45),key='end_charge_time1')
                    SCALE_FACTOR1_CHARGE=st.slider('EV1 charge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
                    battery1_per = st.slider(
                        'Initial EV1 Battery :battery: %',
                        min_value=0.0,
                        max_value=100.0,
                        value=30.0,
                        step=1.0)
                    battery1 =battery1_per / 100 * 300000


                    
            with dis1:
                with st.expander('EV1 DisCharge'):
                    start_discharge_time1 = st.time_input('start discahrge 1 :clock1:',value=time(hour=17, minute=0),key='start_discharge_time1')
                    end_discharge_time1 = st.time_input('end discharge 1:checkered_flag:',value=time(hour=22, minute=45),key='end_discharge_time1')
                    SCALE_FACTOR1_DISHCARGE=st.slider('EV1 discharge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
                    
        with car2:
            st.title('EV2 :car:')
            ch2,dis2=st.columns(2)
            with ch2:
                with st.expander('EV2 Charge'):
                    start_charge_time2 = st.time_input('start charge 2 :clock1:',value=time(hour=8, minute=30))
                    end_charge_time2 = st.time_input('end charge 2 :checkered_flag:',value=time(hour=13, minute=30))
                    SCALE_FUCTOR2_CHARGE=st.slider('EV2 charge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
                    battery2_per = st.slider(
                        'Initial EV2 Battery :battery: %',
                        min_value=0.0,
                        max_value=100.0,
                        value=30.0,
                        step=1.0)
                    battery2 =battery2_per / 100 * 300000
                    
            with dis2:
                with st.expander('EV2 DisCharge'):
                    start_discharge_time2 = st.time_input('start discahrge 2 :clock1:',value=time(hour=19, minute=0))
                    end_discharge_time2 = st.time_input('end discharge 2 :checkered_flag:',value=time(hour=19, minute=30))
                    SCALE_FUCTOR2_DISCHARGE=st.slider('EV2 discharge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
                   
                    
        
        with car3:
            st.title('EV3 :car:')
            ch3,dis3=st.columns(2)
            with ch3:
                with st.expander('EV3 Charge'):
                    start_charge_time3= st.time_input('start charge 3 :clock1:',value=time(hour=9, minute=30))
                    end_charge_time3 = st.time_input('end charge 3 :checkered_flag:',value=time(hour=15, minute=30))
                    SCALE_FUCTOR3_CHARGE=st.slider('EV3 charge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
                    battery3_per = st.slider(
                        'Initial EV3 Battery :battery: %',
                        min_value=0.0,
                        max_value=100.0,
                        value=30.0,
                        step=1.0)
                    battery3 =battery3_per / 100 * 300000
                    
            with dis3:
                with st.expander('EV3 DisCharge'):
                    start_discharge_time3 = st.time_input('start discahrge 3 :clock1:',value=time(hour=17, minute=0))
                    end_discharge_time3 = st.time_input('end discharge 3 :checkered_flag:',value=time(hour=22, minute=15))
                    SCALE_FUCTOR3_DISCHARGE=st.slider('EV3 discharge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
        with car4:
            st.title('EV4 :car:')
            ch4,dis4=st.columns(2)
            with ch4:
                with st.expander('EV4 Charge'):
                    start_charge_time4 = st.time_input('start charge 4 :clock1:',value=time(hour=8, minute=15))
                    end_charge_time4 = st.time_input('end charge 4 :checkered_flag:',value=time(hour=17, minute=45))
                    SCALE_FUCTOR4_CHARGE=st.slider('EV4 charge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
                    battery4_per = st.slider(
                        'Initial EV4 Battery :battery: %',
                        min_value=0.0,
                        max_value=100.0,
                        value=30.0,
                        step=1.0)
                    battery4 =battery4_per / 100 * 300000
                    


            with dis4:
                with st.expander('EV4 DisCharge'):
                    start_discharge_time4 = st.time_input('start discahrge 4 :clock1:',value=time(hour=18, minute=0))
                    end_discharge_time4 = st.time_input('end discharge 4 :checkered_flag:',value=time(hour=21, minute=0))
                    SCALE_FUCTOR4_DISCHARGE=st.slider('EV4 discharge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
        submitted = st.form_submit_button("Submit")
    # Get unique month and day values from the 'day' dataframe
    __get__month = int(day['Month'].unique()[0])  # use index 0 to get the first (and only) element
    __get__day = int(day['DayOfMonth'].unique()[0])

    # Create start and end datetime obkects
    start_date = datetime(year=2021, month=__get__month, day=__get__day)
    end_date = start_date  # set end date equal to start date


    
    merged_df = create_day_charge_profile(day,  [start_charge_time1, start_charge_time2, start_charge_time3, start_charge_time4],
                                                [end_charge_time1, end_charge_time2, end_charge_time3, end_charge_time4],
                                                [SCALE_FACTOR1_CHARGE,SCALE_FUCTOR2_CHARGE,SCALE_FUCTOR3_CHARGE,SCALE_FUCTOR4_CHARGE],
                                                [battery1,battery2,battery3,battery4])
    
    merged_df1 = create_day_discharge_profile(day,  [start_discharge_time1, start_discharge_time2, start_discharge_time3, start_discharge_time4],
                                                    [end_discharge_time1, end_discharge_time2, end_discharge_time3, end_discharge_time4],
                                                    [SCALE_FACTOR1_DISHCARGE,SCALE_FUCTOR2_DISCHARGE,SCALE_FUCTOR3_DISCHARGE,SCALE_FUCTOR4_DISCHARGE])
    
    
    # concatenate the two datasets
    final_profile = pd.concat([merged_df, merged_df1], axis=1)

    # drop duplicate columns
    unique_df = final_profile.loc[:, ~final_profile.columns.duplicated()]
    
    
    
    total_charge,per_car_charge_list=calculateTotalEnergy_EV_Charge(unique_df)
    total_discharge,per_car_discharge_list=calculateTotalEnergy_EV_DisCharge(unique_df)
    with st.container():
        fig, ax = plt.subplots(figsize=(15,4))
        # Create a line plot
        with st.expander('Plot :chart_with_upwards_trend:'):
            with st.form(key='plot'):
                choices=st.multiselect('Select EV',['Total_EV_Charge (W)','EV1_charge (W)', 'EV2_charge (W)', 'EV3_charge (W)', 'EV4_charge (W)', 
                                            'Total_EV_DisCharge (W)','EV1_discharge (W)', 'EV2_discharge (W)', 'EV3_discharge (W)', 'EV4_discharge (W)', 
                                            'PV (W)', 'Total_Imbalance (W)', 'Imbalance (W)'],default=['Total_EV_Charge (W)','EV1_charge (W)', 'EV2_charge (W)', 'EV3_charge (W)', 'EV4_charge (W)', 
                                            'Total_EV_DisCharge (W)','EV1_discharge (W)', 'EV2_discharge (W)', 'EV3_discharge (W)', 'EV4_discharge (W)', 
                                            'PV (W)', 'Total_Imbalance (W)', 'Imbalance (W)'])
                submitted = st.form_submit_button("Submit")
            unique_df['Total_Imbalance (W)']=unique_df['Imbalance (W)']+unique_df['Total_EV_Charge (W)']+unique_df['Total_EV_DisCharge (W)']
            sns.lineplot(data=unique_df[choices],
                                        palette=['#000b5e','#021496','#0119cb','#001be7','#001eff',
                                                '#ff0000','#ff7100','#ffa900','#ffcb00','#fff800',  
                                                '#00FF00','#ff00e1','#ad00ff'],ax=ax)
            
            plt.title('Electric Vehicle Charging')
            plt.xlabel('Time')
            plt.ylabel('Charge (W)')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            plt.legend(loc='lower left',fontsize='x-small')
            st.pyplot(fig)
    with st.container():
        with st.expander('Daily Energy Exchange Analysis :clipboard:'):
            col1,col2,col3=st.columns([2,1,0.9])
            with col1:
                fig1, ax1 = plt.subplots()
                plt.bar(range(len(per_car_charge_list)), per_car_charge_list, color='b')
                # plot bar chart with discharge values
                plt.bar(range(len(per_car_discharge_list)), per_car_discharge_list, color='r')
                # set x-axis labels
                plt.xticks(range(len(per_car_charge_list)), ['Car 1', 'Car 2', 'Car 3', 'Car 4'])
                ax1.text(0, per_car_charge_list[0], str(round(per_car_charge_list[0])), ha='center', va='bottom')
                ax1.text(0, per_car_discharge_list[0], str(round(per_car_discharge_list[0])), ha='center', va='bottom')

                ax1.text(1, per_car_charge_list[1], str(round(per_car_charge_list[1])), ha='center', va='bottom')
                ax1.text(1, per_car_discharge_list[1], str(round(per_car_discharge_list[1])), ha='center', va='bottom')

                ax1.text(2, per_car_charge_list[2], str(round(per_car_charge_list[2])), ha='center', va='bottom')
                ax1.text(2, per_car_discharge_list[2], str(round(per_car_discharge_list[2])), ha='center', va='bottom')

                ax1.text(3, per_car_charge_list[3], str(round(per_car_charge_list[3])), ha='center', va='bottom')
                ax1.text(3, per_car_discharge_list[3], str(round(per_car_discharge_list[3])), ha='center', va='bottom')

                # set y-axis label
                plt.ylabel('Energy (kWh)')
                # set title
                plt.title('Total Energy Exchange for each EVs')
                # show plot
                st.pyplot(fig1)
            with col2:
                fig2, ax2= plt.subplots(figsize=(4, 7))
                plt.bar(['Total Discharge', 'Total Charge'], [total_discharge, total_charge], color=['r', 'b'])

                ax2.text(1, total_charge, str(round(total_charge,2)), ha='center', va='bottom')
                ax2.text(0, total_discharge, str(round(total_discharge,2)), ha='center', va='bottom')
                plt.ylabel('Energy (kWh)')
                plt.title('Total Energy exchange all EV')
                st.pyplot(fig2)
                
                good_energy_count,bad_energy_count,total_EV_demand_count,per_car_count_list=count_positive_charge_negative_imbalance(unique_df)
            with col3:
                st.pyplot(plot_pie_chart(["PVs","From Grid"],[good_energy_count,bad_energy_count]))
            
        starting_car_capacity_list=[battery1,battery2,battery3,battery4]
        calculate_energy_storage(unique_df,4,starting_car_capacity_list)
        
            
            
        with st.expander(':battery: Battery 1'):
            c11,c12,c13 = st.columns([1,1,3])
            with c11:
                st.subheader('Start')
                car1_battery_lvl1 = unique_df['BatteryLVL1'][1]
                car1_html1=HtmlGenerator.battery_lvl1(1,car1_battery_lvl1)
                st.markdown(car1_html1, unsafe_allow_html=True)
            
                st.subheader('End')
                car1_battery_lvl11 = unique_df['BatteryLVL1'][-1]
                car1_html11=HtmlGenerator.battery_lvl1(1,car1_battery_lvl11)
                st.markdown(car1_html11, unsafe_allow_html=True)
            with c12:
                try:
                    fig12,ax12=plt.subplots(figsize=(1,1))
                    count_g1=unique_df[(unique_df['EV1_charge (W)'] > 0) & (unique_df['Total_Imbalance (W)'] < 0)]['BatteryLVL1'].count()
                    count_1=unique_df[(unique_df['EV1_charge (W)'] > 0) & (unique_df['Total_Imbalance (W)'] > 0)]['BatteryLVL1'].count()
                    fig22=(plot_pie_chart(['Green','Red'],[count_g1,count_1]))
                    st.pyplot(fig22)
                except:
                    st.markdown('Neither charging nor discharging happened')
            with c13:
                fig,ax=plt.subplots(figsize=(10,2))
                ax.stackplot(unique_df.index, unique_df['BatteryLVL1'], color='r')
                unique_df['EV1_green_energy'] = unique_df[(unique_df['EV1_charge (W)'] > 0) & (unique_df['Total_Imbalance (W)'] < 0)]['BatteryLVL1']
                
                ax.stackplot(unique_df.index, unique_df['EV1_green_energy'], color='g')
                ax.axhline(y=MAXIMUM_CAR_CAPACITY, xmin=0, xmax=1, color='blue', linestyle='-',label='Maximum Capacity')
                ax.axhline(y=MAXIMUM_CAR_CAPACITY*0.8, xmin=0, xmax=1, color='blue', linestyle='--',label='80% Capacity%')
                ax.axhline(y=MAXIMUM_CAR_CAPACITY*0.2, xmin=0, xmax=1, color='blue', linestyle='--',label='20% Capacity')
                ax.axhline(y=0, xmin=0, xmax=1, color='blue', linestyle='--',label='Minimum Capacity')
                ax.set_title(f'EV{1}')
                st.pyplot(fig)
                
                st.write('------------------------------------')
        with st.expander(':battery: Battery 2'):
            c21,c22,c23 = st.columns([1,1,3])
            with c21:
                st.subheader('Start')
                car1_battery_lvl2 = unique_df['BatteryLVL2'][1]
                car1_html2=HtmlGenerator.battery_lvl1(2,car1_battery_lvl2)
                st.markdown(car1_html2, unsafe_allow_html=True)
            
                st.subheader('End')
                car1_battery_lvl12 = unique_df['BatteryLVL2'][-1]
                car1_html12=HtmlGenerator.battery_lvl1(1,car1_battery_lvl12)
                st.markdown(car1_html12, unsafe_allow_html=True)
            with c22:
                try:   
                    fig22,ax22=plt.subplots(figsize=(1,1))
                    count_g2=unique_df[(unique_df['EV2_charge (W)'] > 0) & (unique_df['Total_Imbalance (W)'] < 0)]['BatteryLVL2'].count()
                    count_2=unique_df[(unique_df['EV2_charge (W)'] > 0) & (unique_df['Total_Imbalance (W)'] > 0)]['BatteryLVL2'].count()
                    fig22=(plot_pie_chart(['Green','Red'],[count_g2,count_2]))
                    st.pyplot(fig22)
                except:
                    st.markdown('Neither charging nor discharging happened')
            with c23:
                fig,ax=plt.subplots(figsize=(10,2))
                ax.stackplot(unique_df.index, unique_df['BatteryLVL2'], color='r')
                unique_df['EV2_green_energy'] = unique_df[(unique_df['EV2_charge (W)'] > 0) & (unique_df['Total_Imbalance (W)'] < 0)]['BatteryLVL2']
                ax.stackplot(unique_df.index, unique_df['EV2_green_energy'], color='g')
                ax.axhline(y=MAXIMUM_CAR_CAPACITY, xmin=0, xmax=1, color='blue', linestyle='-',label='Maximum Capacity')
                ax.axhline(y=MAXIMUM_CAR_CAPACITY*0.8, xmin=0, xmax=1, color='blue', linestyle='--',label='80% Capacity%')
                ax.axhline(y=MAXIMUM_CAR_CAPACITY*0.2, xmin=0, xmax=1, color='blue', linestyle='--',label='20% Capacity')
                ax.axhline(y=0, xmin=0, xmax=1, color='blue', linestyle='--',label='Minimum Capacity')
                ax.set_title(f'EV{2}')
                st.pyplot(fig) 
            st.write('------------------------------------')
        with st.expander(':battery: Battery 3'):
            c31,c32,c33 = st.columns([1,1,3])
            with c31:
                st.subheader('Start')
                car1_battery_lvl3 = unique_df['BatteryLVL3'][1]
                car1_html3=HtmlGenerator.battery_lvl1(1,car1_battery_lvl3)
                st.markdown(car1_html3, unsafe_allow_html=True)
            
                st.subheader('End')
                car1_battery_lvl13 = unique_df['BatteryLVL3'][-1]
                car1_html13=HtmlGenerator.battery_lvl1(1,car1_battery_lvl13)
                st.markdown(car1_html13, unsafe_allow_html=True)
            with c32:
                try:
                    fig32,ax32=plt.subplots(figsize=(1,1))
                    count_g3=unique_df[(unique_df['EV3_charge (W)'] > 0) & (unique_df['Total_Imbalance (W)'] < 0)]['BatteryLVL3'].count()
                    count_3=unique_df[(unique_df['EV3_charge (W)'] > 0) & (unique_df['Total_Imbalance (W)'] > 0)]['BatteryLVL3'].count()
                    fig32=(plot_pie_chart(['Green','Red'],[count_g3,count_3]))
                    st.pyplot(fig32)
                except:
                    st.markdown('Neither charging nor discharging happened')
            with c33:
                fig,ax=plt.subplots(figsize=(10,2))
                ax.stackplot(unique_df.index, unique_df['BatteryLVL3'], color='r')
                unique_df['EV3_green_energy'] = unique_df[(unique_df['EV3_charge (W)'] > 0) & (unique_df['Total_Imbalance (W)'] < 0)]['BatteryLVL3']
                ax.stackplot(unique_df.index, unique_df['EV3_green_energy'], color='g')
                ax.axhline(y=MAXIMUM_CAR_CAPACITY, xmin=0, xmax=1, color='r', linestyle='-',label='Maximum Capacity')
                ax.axhline(y=MAXIMUM_CAR_CAPACITY*0.8, xmin=0, xmax=1, color='blue', linestyle='--',label='80% Capacity%')
                ax.axhline(y=MAXIMUM_CAR_CAPACITY*0.2, xmin=0, xmax=1, color='blue', linestyle='--',label='20% Capacity')
                ax.axhline(y=0, xmin=0, xmax=1, color='blue', linestyle='--',label='Minimum Capacity')
                ax.set_title(f'EV{3}')
                st.pyplot(fig) 
            st.write('------------------------------------')
        with st.expander(':battery: Battery 4'):
            c41,c42,c43 = st.columns([1,1,3])
            with c41:
                st.subheader('Start')
                car1_battery_lvl1 = unique_df['BatteryLVL4'][1]
                car1_html1=HtmlGenerator.battery_lvl1(1,car1_battery_lvl1)
                st.markdown(car1_html1, unsafe_allow_html=True)
            
                st.subheader('End')
                car1_battery_lvl11 = unique_df['BatteryLVL4'][-1]
                car1_html11=HtmlGenerator.battery_lvl1(1,car1_battery_lvl11)
                st.markdown(car1_html11, unsafe_allow_html=True)
            with c42:
                try:
                    fig42,ax42=plt.subplots(figsize=(1,1))
                    count_g4=unique_df[(unique_df['EV4_charge (W)'] > 0) & (unique_df['Total_Imbalance (W)'] < 0)]['BatteryLVL4'].count()
                    count_4=unique_df[(unique_df['EV4_charge (W)'] > 0) & (unique_df['Total_Imbalance (W)'] > 0)]['BatteryLVL4'].count()
                    fig42=(plot_pie_chart(['Green','Red'],[count_g4,count_4]))
                    st.pyplot(fig42)
                except:
                    st.markdown('Neither charging nor discharging happened')
            with c43:
                fig,ax=plt.subplots(figsize=(10,3))
                ax.stackplot(unique_df.index, unique_df['BatteryLVL4'], color='r')
                unique_df['EV4_green_energy'] = unique_df[(unique_df['EV4_charge (W)'] > 0) & (unique_df['Total_Imbalance (W)'] < 0)]['BatteryLVL4']
                ax.stackplot(unique_df.index, unique_df['EV4_green_energy'], color='g')
                ax.axhline(y=MAXIMUM_CAR_CAPACITY, xmin=0, xmax=1, color='blue', linestyle='-',label='Maximum Capacity')
                ax.axhline(y=MAXIMUM_CAR_CAPACITY*0.8, xmin=0, xmax=1, color='blue', linestyle='--',label='80% Capacity%')
                ax.axhline(y=MAXIMUM_CAR_CAPACITY*0.2, xmin=0, xmax=1, color='blue', linestyle='--',label='20% Capacity')
                ax.axhline(y=0, xmin=0, xmax=1, color='blue', linestyle='--',label='Minimum Capacity')
                ax.set_title(f'EV{4}')
                
                
                st.pyplot(fig) 
                
            st.write('------------------------------------')
        st.download_button(
            label="Download data as CSV",
            data=unique_df.to_csv(index=True),
            file_name=f'{DAY}.csv',
            mime='text/csv')
            
            
            
        
        
if __name__ == '__main__':
     
     sns.set(style="darkgrid")
     st. set_page_config(layout="wide")
     st.title('EV Charging and Discharging Planner for a day')

     days_mapping = {}
     day_index = 0
     for day in range(1, 323):
        day_index += 1
        if day_index % 31 == 0:
            day_index = 1
        formatted_date = f"{day_index}/{calendar.month_abbr[(day - 1) // 27 + 1]}"
        days_mapping[day] = formatted_date

     
     DAY = st.selectbox('Select Day :calendar:', list(days_mapping.keys()), format_func=lambda key: days_mapping[key])

     MAXIMUM_CAR_CAPACITY = st.number_input(
                ':battery: Capacity in kWh',
                min_value=0,
                max_value=int(300000*0.00025),
                value=int(300000*0.00025),
                step=int(10000*0.00025),
                key="max_car_cap_input"
            ) / 0.00025
     st.sidebar.image('imgs/SaxionLogo.png',width=200)
     
     st.sidebar.markdown('**Designed By </br> Stelios Diamantopoulos**',unsafe_allow_html=True)
     main()
     