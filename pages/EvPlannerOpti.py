import pandas as pd
import numpy as np
from datetime import time,timedelta,datetime
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import itertools
NO_CARS=2

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
    df['PV (W)']=df['PV (W)']*327
    df['TotalDemand (W)']=df['General Demand (W)']+df['Heating Demand (W)']
    df['Imbalance (W)']=df['TotalDemand (W)']+df['PV (W)']
    power_imported = np.where(df['Imbalance (W)'] > 0, df['TotalDemand (W)']+df['PV (W)'],0)
    power_wasted = np.where(df['Imbalance (W)'] < 0, df['TotalDemand (W)']+df['PV (W)'],0)
    df['Power Imported (W)'] = power_imported
    df['Power Wasted (W)'] = power_wasted
    days=split_dataframe_by_day(df)
    
    return days


class ProfileGenerator:
    def crete_charge_profile(df,start_time,end_time,start_date,end_date,id,scale_factor) :
        print('TYPEEEE',type(start_time), start_time)
        date_range = pd.date_range(start=datetime.combine(start_date, start_time), end=datetime.combine(end_date, end_time), freq='15min')
        charge_profile = pd.DataFrame(index=date_range)
        charge_profile[f'EV{id}_charge (W)'] = pd.Series(data=scale_factor*np.random.randint(low=6400, high=7000, size=len(date_range)), index=charge_profile.index)
        charge_profile.index.name = 'Time'
        
        return charge_profile
    
    def crete_discharge_profile(df,start_time,end_time,start_date,end_date,id,scale_factor):
        date_range = pd.date_range(start=datetime.combine(start_date, start_time), end=datetime.combine(end_date, end_time), freq='15min')
        discharge_profile = pd.DataFrame(index=date_range)
        discharge_profile[f'EV{id}_discharge (W)'] = pd.Series(data=scale_factor*np.random.randint(low=-7000, high=-6400, size=len(date_range)), index=discharge_profile.index)
        discharge_profile.index.name = 'Time'
        
        return discharge_profile


class MergeProfiles:
    def merge_charge_profile(df,charge_profile):
        merged_df = pd.concat([charge_profile.set_index('Time'), df.set_index('Time')], axis=1)
        merged_df.fillna(0, inplace=True)
        merged_df.drop_duplicates(inplace=True)
        
        
        # print(len(charge_profile))
        # print(charge_profile,'CHARGE PROFILE')
        # print(type(charge_profile),'YTPE')

        
        # frame = charge_profile.sum(axis=1)

        #frame = charge_profile.sum(axis=1).to_frame()
        
        # frame.rename(columns = {0:'TOTAL_EV_CHARGE'}, inplace = True)
        
        #print(frame,'FRAME')

        #merged_df['Total_EV_CHARGE'] = frame
        
        merged_df['Total_EV_Charge (W)']=merged_df.apply(lambda x: x[0]+x[1],axis=1)
        
        # print(merged_df.head(50))

        
        merged_df['TotalImbalance']=merged_df['Imbalance (W)']+merged_df['Total_EV_Charge (W)']
        
        return merged_df
    
    def merge_discharge_profile(df,discharge_profile):
        merged_df = pd.concat([discharge_profile.set_index('Time'), df.set_index('Time')], axis=1)
        merged_df.fillna(0, inplace=True)
        merged_df.drop_duplicates(inplace=True)
        print(merged_df.columns,"HERE")
        merged_df['Total_EV_Discharge (W)']=merged_df.apply(lambda x: x[0]+x[1],axis=1)
        merged_df['TotalImbalance']=merged_df['Imbalance (W)'] + merged_df['Total_EV_Discharge (W)']
        #print(merged_df,"DIS")

        return merged_df

    

def create_day_charge_profile(day, start_charge_times, end_charge_times,SCALE_FACTORS_CHARGE):
    # Get unique month and day values from the 'day' dataframe
    __get__month = int(day['Month'].unique()[0])  # use index 0 to get the first (and only) element
    __get__day = int(day['DayOfMonth'].unique()[0])

    # Create start and end datetime objects
    start_date = datetime(year=2021, month=__get__month, day=__get__day)
    end_date = start_date  # set end date equal to start date
    
    evArray = []
    
    NO_CARS = len(start_charge_times)
    
    for x in range(0,len(start_charge_times)):
        # Merge the charge profiles into a single dataframe
        evArray.append(ProfileGenerator.crete_charge_profile(day, start_charge_times[x], end_charge_times[x], start_date, end_date, id=x+1, scale_factor=SCALE_FACTORS_CHARGE[x]))
        #print(evArray)
        
    charge_data = pd.concat(evArray, axis=1)
    charge_data.reset_index(inplace=True)
    charge_data.fillna(0, inplace=True)
        

    # Merge the charge profile with the original dataframe
    merged_df = MergeProfiles.merge_charge_profile(day, charge_data)
    
    return merged_df
def create_day_discharge_profile(day, start_charge_times, end_charge_times,SCALE_FACTORS_DISCHARGE):
    # Get unique month and day values from the 'day' dataframe
    __get__month = int(day['Month'].unique()[0])  # use index 0 to get the first (and only) element
    __get__day = int(day['DayOfMonth'].unique()[0])

    # Create start and end datetime objects
    start_date = datetime(year=2021, month=__get__month, day=__get__day)
    end_date = start_date  # set end date equal to start date
    
    
    evArray= []
    for x in range(0,len(start_charge_times)):
            evArray.append(ProfileGenerator.crete_discharge_profile(day, start_charge_times[x], end_charge_times[x], start_date, end_date, id=x+1, scale_factor=SCALE_FACTORS_DISCHARGE[x]))

        # Merge the charge profiles into a single dataframe
    discharge_data = pd.concat(evArray, axis=1)
    discharge_data.reset_index(inplace=True)
    discharge_data.fillna(0, inplace=True)

    # Merge the charge profile with the original dataframe
    merged_df = MergeProfiles.merge_discharge_profile(day, discharge_data)
    
    return merged_df
def calculateTotalEnergy_EV_Charge(df):
    population=df['Total_EV_Charge (W)'].sum()
    try:
        cars=[]                           
        for x in range(0,NO_CARS) :
            cars.append(df[x].sum())
            
        print(cars,'CARS')
        
    except:
        pass
    return population , cars

def calculateTotalEnergy_EV_DisCharge(df):
    print(df.columns)
    population=df['Total_EV_Discharge (W)'].sum()
    
    try:
        cars=[]                           
        for x in range(0,NO_CARS) :
            cars.append(df[x].sum())
            
        print(cars,'CARS')
        
    except:
        pass
    return population , cars

def count_positive_charge_negative_imbalance(df):
    count=0
    count1=0
    count  += len(df[(df['EV1_charge (W)'] > 0) & (df['TotalImbalance'] <= 0)])
    count  += len(df[(df['EV2_charge (W)'] > 0) & (df['TotalImbalance'] <= 0)])

    count1 += len(df[(df['EV1_charge (W)'] > 0) & (df['TotalImbalance'] >= 0)])
    count1 += len(df[(df['EV2_charge (W)'] > 0) & (df['TotalImbalance'] >= 0)])

    total_count=len(df['TotalImbalance']>0)
    

    
    return count,count1 ,total_count, [len(df[(df['EV1_charge (W)'] > 0) & (df['TotalImbalance'] <= 0)]),
                                       len(df[(df['EV2_charge (W)'] > 0) & (df['TotalImbalance'] <= 0)]),
                                       ]
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
    fig, ax = plt.subplots(figsize = (4,4))
    plt.title('Energy Origin Perchentage used for charging EVs')
    ax.pie(values, labels=labels, autopct='%1.1f%%',colors=('g','r'))
    
    ax.set_aspect('equal')
    plt.show()
    return fig

def main():
    
    days=ETL()
    day=days[DAY]
    with st.form("my_form"):
        car1,car2,car3,car4=st.columns(4)
        with car1:
            st.title('EV1')
            ch1,dis1=st.columns(2)
            with ch1:
                with st.expander('EV1 Charge'):
                    start_charge_time1 = st.time_input('start charge time 1',value=time(hour=9, minute=30),key='start_charge_time1')
                    st.write(type(start_charge_time1))
                    st.write(start_charge_time1)

                    end_charge_time1 = st.time_input('end charge time 1',value=time(hour=15, minute=45),key='end_charge_time1')
                    SCALE_FACTOR1_CHARGE=st.slider('EV1 charge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
                    
            with dis1:
                with st.expander('EV1 DisCharge'):
                    start_discharge_time1 = st.time_input('start discahrge time 1',value=time(hour=17, minute=0),key='start_discharge_time1')
                    end_discharge_time1 = st.time_input('end discharge time 1',value=time(hour=22, minute=45),key='end_discharge_time1')
                    SCALE_FACTOR1_DISHCARGE=st.slider('EV1 discharge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
        with car2:
            st.title('EV2')
            ch2,dis2=st.columns(2)
            with ch2:
                with st.expander('EV2 Charge'):
                    start_charge_time2 = st.time_input('start charge time 2',value=time(hour=8, minute=30))
                    end_charge_time2 = st.time_input('end charge time 2',value=time(hour=18, minute=0))
                    SCALE_FUCTOR2_CHARGE=st.slider('EV2 charge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
                    
            with dis2:
                with st.expander('EV2 DisCharge'):
                    start_discharge_time2 = st.time_input('start discahrge time 2',value=time(hour=19, minute=0))
                    end_discharge_time2 = st.time_input('end discharge time 2',value=time(hour=19, minute=30))
                    SCALE_FUCTOR2_DISCHARGE=st.slider('EV2 discharge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
        
        with car3:
            st.title('EV3')
            ch3,dis3=st.columns(2)
            with ch3:
                with st.expander('EV3 Charge'):
                    start_charge_time3= st.time_input('start charge time 3',value=time(hour=9, minute=30))
                    end_charge_time3 = st.time_input('end charge time 3',value=time(hour=15, minute=30))
                    SCALE_FUCTOR3_CHARGE=st.slider('EV3 charge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
                    
            with dis3:
                with st.expander('EV3 DisCharge'):
                    start_discharge_time3 = st.time_input('start discahrge time 3',value=time(hour=17, minute=0))
                    end_discharge_time3 = st.time_input('end discharge time 3',value=time(hour=22, minute=15))
                    SCALE_FUCTOR3_DISCHARGE=st.slider('EV3 discharge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
        with car4:
            st.title('EV4')
            ch4,dis4=st.columns(2)
            with ch4:
                with st.expander('EV4 Charge'):
                    start_charge_time4 = st.time_input('start charge time 4',value=time(hour=8, minute=15))
                    end_charge_time4 = st.time_input('end charge time 4',value=time(hour=17, minute=45))
                    SCALE_FUCTOR4_CHARGE=st.slider('EV4 charge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
            with dis4:
                with st.expander('EV4 DisCharge'):
                    start_discharge_time4 = st.time_input('start discahrge time 4',value=time(hour=18, minute=0))
                    end_discharge_time4 = st.time_input('end discharge time 4',value=time(hour=21, minute=0))
                    SCALE_FUCTOR4_DISCHARGE=st.slider('EV4 discharge multiplier',min_value=0.0,max_value=5.0,value=1.0,step=0.1)
        submitted = st.form_submit_button("Submit")
    # Get unique month and day values from the 'day' dataframe
    __get__month = int(day['Month'].unique()[0])  # use index 0 to get the first (and only) element
    __get__day = int(day['DayOfMonth'].unique()[0])

    # Create start and end datetime objects
    start_date = datetime(year=2021, month=__get__month, day=__get__day)
    end_date = start_date  # set end date equal to start date
    
    
    #CREATE OUR OWN DATASET TO ITERATE
    
    #Datetime
    date_cars = []
    
    
        
        
    EV1_charge_start =  time(9,0)
    EV1_charge_end =  time(15,0)
    
    EV2_charge_start = time(18,0)
    EV2_charge_end = time(20,0)
    
    
    ev1 = pd.date_range(start=datetime.combine(start_date, EV1_charge_start), end=datetime.combine(end_date, EV1_charge_end), freq='15min').time
    ev2 = pd.date_range(start=datetime.combine(start_date, EV2_charge_start), end=datetime.combine(end_date, EV2_charge_end), freq='15min').time
    date_cars=[ev1,ev2]
    
    car1 = []
    car2 = []
    # Iterate through all possible combinations of 3 times in ev1
    for r1 in range(len(ev1) + 1):
        for combination1 in itertools.combinations(ev1, r1):
            if len(combination1) == 2:
                car1.append(combination1)

    # Iterate through all possible combinations of 3 times in ev2
    for r2 in range(len(ev2) + 1):
        for combination2 in itertools.combinations(ev2, r2):
            if len(combination2) == 2:
                car2.append(combination2)

    # Print the results
    
    st.write(car1)
    st.write(car2)
        
            
    #HERE##
            
                # merged_df = create_day_charge_profile(day,  [element[i],element[i]],
                #                                             [end_charge_time1,end_charge_time2],
                #                                             [SCALE_FACTOR1_CHARGE,SCALE_FUCTOR2_CHARGE])
                

                # merged_df1 = create_day_discharge_profile(day,  [start_discharge_time1,start_discharge_time2],
                #                                                 [end_discharge_time1,end_discharge_time2],
                #                                                 [SCALE_FACTOR1_DISHCARGE,SCALE_FUCTOR2_DISCHARGE])
                
                
                # # concatenate the two datasets
                # final_profile = pd.concat([merged_df, merged_df1], axis=1)

                # # drop duplicate columns
                # unique_df = final_profile.loc[:, ~final_profile.columns.duplicated()]


                # total_charge,per_car_charge_list=calculateTotalEnergy_EV_Charge(unique_df)
                # total_discharge,per_car_discharge_list=calculateTotalEnergy_EV_DisCharge(unique_df)

            

                # good_energy_count,bad_energy_count,total_EV_demand_count,per_car_count_list=count_positive_charge_negative_imbalance(unique_df)
                
                # results.append([bad_energy_count,good_energy_count,element[1],element[2]])
                # #st.pyplot(plot_pie_chart(["PVs","From Grid"],[good_energy_count,bad_energy_count]))

                            
            
    
        



        
if __name__ == '__main__':
     
     sns.set(style="darkgrid")
     st. set_page_config(layout="wide")
     st.title('EV Charging and Discharging Planner for a day')
     DAY=st.selectbox('Select Day',range(1,322))
     main()