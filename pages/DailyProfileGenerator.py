import pandas as pd
import numpy as np
from datetime import time,timedelta,datetime
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
DAY=223
SCALE_FACTOR=5

def split_dataframe_by_day(df):
        days = [df[i:i+96] for i in range(0, len(df), 96)]
        return days


def ETL():
    df = pd.read_csv('../data_original.csv')
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
        date_range = pd.date_range(start=datetime.combine(start_date, start_time), end=datetime.combine(end_date, end_time), freq='15min')
        charge_profile = pd.DataFrame(index=date_range)
        charge_profile[f'EV{id}_charge (W)'] = pd.Series(data=scale_factor*np.random.randint(low=6400, high=7000, size=len(date_range)), index=charge_profile.index)
        charge_profile.index.name = 'Time'
        
        return charge_profile
    
    def crete_discharge_profile(df,start_time,end_time,start_date,end_date,id):
        date_range = pd.date_range(start=datetime.combine(start_date, start_time), end=datetime.combine(end_date, end_time), freq='15min')
        discharge_profile = pd.DataFrame(index=date_range)
        discharge_profile[f'EV{id}_charge (W)'] = pd.Series(data=np.random.randint(low=-7000, high=-6400, size=len(date_range)), index=discharge_profile.index)
        discharge_profile.index.name = 'Time'
        
        return discharge_profile


class MergeProfiles:
    def merge_charge_profile(df,charge_profile):
        merged_df = pd.concat([charge_profile.set_index('Time'), df.set_index('Time')], axis=1)
        merged_df.fillna(0, inplace=True)
        merged_df.drop_duplicates(inplace=True)
        
        merged_df['TotalImbalance']=merged_df['Imbalance (W)']+merged_df['EV1_charge (W)']+merged_df['EV2_charge (W)']+merged_df['EV3_charge (W)']+merged_df['EV4_charge (W)']
        merged_df['Total_EV_Charge (W)']=merged_df['EV1_charge (W)']+merged_df['EV2_charge (W)']+merged_df['EV3_charge (W)']+merged_df['EV4_charge (W)']
        return merged_df
    
    def merge_discharge_profile(df,discharge_profile):
        merged_df = pd.concat([discharge_profile.set_index('Time'), df.set_index('Time')], axis=1)
        merged_df.fillna(0, inplace=True)
        merged_df.drop_duplicates(inplace=True)
        
        merged_df['TotalImbalance']=merged_df['Imbalance (W)']+merged_df['EV1_charge (W)']+merged_df['EV2_charge (W)']+merged_df['EV3_charge (W)']+merged_df['EV4_charge (W)']
        merged_df['Total_EV_Charge (W)']=merged_df['EV1_charge (W)']+merged_df['EV2_charge (W)']+merged_df['EV3_charge (W)']+merged_df['EV4_charge (W)']
        return merged_df
    

def create_day_charge_profile(day, start_charge_times, end_charge_times,SCALE_FACTORS_CHARGE):
    # Get unique month and day values from the 'day' dataframe
    __get__month = int(day['Month'].unique()[0])  # use index 0 to get the first (and only) element
    __get__day = int(day['DayOfMonth'].unique()[0])

    # Create start and end datetime objects
    start_date = datetime(year=2021, month=__get__month, day=__get__day)
    end_date = start_date  # set end date equal to start date

    # Create charge profiles for all four EVs
    ev1 = ProfileGenerator.crete_charge_profile(day, start_charge_times[0], end_charge_times[0], start_date, end_date, id=1, scale_factor=SCALE_FACTORS_CHARGE[0])
    ev2 = ProfileGenerator.crete_charge_profile(day, start_charge_times[1], end_charge_times[1], start_date, end_date, id=2, scale_factor=SCALE_FACTORS_CHARGE[1])
    ev3 = ProfileGenerator.crete_charge_profile(day, start_charge_times[2], end_charge_times[2], start_date, end_date, id=3, scale_factor=SCALE_FACTORS_CHARGE[2])
    ev4 = ProfileGenerator.crete_charge_profile(day, start_charge_times[3], end_charge_times[3], start_date, end_date, id=4, scale_factor=SCALE_FACTORS_CHARGE[3])

    # Merge the charge profiles into a single dataframe
    charge_data = pd.concat([ev1, ev2, ev3, ev4], axis=1)
    charge_data.reset_index(inplace=True)
    charge_data.fillna(0, inplace=True)

    # Merge the charge profile with the original dataframe
    merged_df = MergeProfiles.merge_charge_profile(day, charge_data)
    
    return merged_df
def create_day_discharge_profile(day, start_charge_times, end_charge_times,SCALE_FACTORS):
    # Get unique month and day values from the 'day' dataframe
    __get__month = int(day['Month'].unique()[0])  # use index 0 to get the first (and only) element
    __get__day = int(day['DayOfMonth'].unique()[0])

    # Create start and end datetime objects
    start_date = datetime(year=2021, month=__get__month, day=__get__day)
    end_date = start_date  # set end date equal to start date

    # Create charge profiles for all four EVs
    ev1 = ProfileGenerator.crete_charge_profile(day, start_charge_times[0], end_charge_times[0], start_date, end_date, id=1,scale_factor=SCALE_FACTORS[0])
    ev2 = ProfileGenerator.crete_charge_profile(day, start_charge_times[1], end_charge_times[1], start_date, end_date, id=2,scale_factor=SCALE_FACTORS[1])
    ev3 = ProfileGenerator.crete_charge_profile(day, start_charge_times[2], end_charge_times[2], start_date, end_date, id=3,scale_factor=SCALE_FACTORS[2])
    ev4 = ProfileGenerator.crete_charge_profile(day, start_charge_times[3], end_charge_times[3], start_date, end_date, id=4,scale_factor=SCALE_FACTORS[3])

    # Merge the charge profiles into a single dataframe
    charge_data = pd.concat([ev1, ev2, ev3, ev4], axis=1)
    charge_data.reset_index(inplace=True)
    charge_data.fillna(0, inplace=True)

    # Merge the charge profile with the original dataframe
    merged_df = MergeProfiles.merge_charge_profile(day, charge_data)
    
    return merged_df
def calculateTotalEnergy_EV_Charge(df):
    population = 0
    population += df['EV1_charge (W)'].sum()
    population += df['EV2_charge (W)'].sum()
    population += df['EV3_charge (W)'].sum()
    population += df['EV4_charge (W)'].sum()
    return population , [df['EV1_charge (W)'].sum(),df['EV2_charge (W)'].sum(),df['EV3_charge (W)'].sum(),df['EV4_charge (W)'].sum()]
def count_positive_charge_negative_imbalance(df):
    count=0
    count1=0
    count += len(df[(df['EV1_charge (W)'] > 0) & (df['TotalImbalance'] < 0)])
    count += len(df[(df['EV2_charge (W)'] > 0) & (df['TotalImbalance'] < 0)])
    count += len(df[(df['EV3_charge (W)'] > 0) & (df['TotalImbalance'] < 0)])
    count += len(df[(df['EV4_charge (W)'] > 0) & (df['TotalImbalance'] < 0)])
    count1 += len(df[(df['EV1_charge (W)'] > 0) & (df['TotalImbalance'] > 0)])
    count1 += len(df[(df['EV2_charge (W)'] > 0) & (df['TotalImbalance'] > 0)])
    count1 += len(df[(df['EV3_charge (W)'] > 0) & (df['TotalImbalance'] > 0)])
    count1 += len(df[(df['EV4_charge (W)'] > 0) & (df['TotalImbalance'] > 0)])
    total_count=len(df['EV1_charge (W)']>0)


    
    return count,count1 ,total_count, [len(df[(df['EV1_charge (W)'] > 0) & (df['TotalImbalance'] < 0)]),
                                len(df[(df['EV2_charge (W)'] > 0) & (df['TotalImbalance'] < 0)]),
                                len(df[(df['EV3_charge (W)'] > 0) & (df['TotalImbalance'] < 0)]),
                                len(df[(df['EV4_charge (W)'] > 0) & (df['TotalImbalance'] < 0)])]
  

def create_day_discharge_profile(day, start_discharge_times, end_discharge_times,SCALE_FACTORS):
    # Get unique month and day values from the 'day' dataframe
    __get__month = int(day['Month'].unique()[0])  # use index 0 to get the first (and only) element
    __get__day = int(day['DayOfMonth'].unique()[0])

    # Create start and end datetime objects
    start_date = datetime(year=2021, month=__get__month, day=__get__day)
    end_date = start_date  # set end date equal to start date

    # Create charge profiles for all four EVs
    ev1 = ProfileGenerator.crete_discharge_profile(day, start_discharge_times[0], end_discharge_times[0], start_date, end_date, id=1)
    ev2 = ProfileGenerator.crete_discharge_profile(day, start_discharge_times[1], end_discharge_times[1], start_date, end_date, id=2)
    ev3 = ProfileGenerator.crete_discharge_profile(day, start_discharge_times[2], end_discharge_times[2], start_date, end_date, id=3)
    ev4 = ProfileGenerator.crete_discharge_profile(day, start_discharge_times[3], end_discharge_times[3], start_date, end_date, id=4)

    # Merge the charge profiles into a single dataframe
    charge_data = pd.concat([ev1, ev2, ev3, ev4], axis=1)
    charge_data.reset_index(inplace=True)
    charge_data.fillna(0, inplace=True)

    # Merge the charge profile with the original dataframe
    merged_df = MergeProfiles.merge_charge_profile(day, charge_data)
    
    return merged_df

def plot_pie_chart(labels, values):
    fig, ax = plt.subplots()
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



    merged_df = create_day_charge_profile(day, [start_charge_time1, start_charge_time2, start_charge_time3, start_charge_time4],
                                                [end_charge_time1, end_charge_time2, end_charge_time3, end_charge_time4],[SCALE_FACTOR1_CHARGE,SCALE_FUCTOR2_CHARGE,SCALE_FUCTOR3_CHARGE,SCALE_FUCTOR4_CHARGE])

    merged_df1 = create_day_discharge_profile(day, [start_discharge_time1, start_discharge_time2, start_discharge_time3, start_discharge_time4],
                                                    [end_discharge_time1, end_discharge_time2, end_discharge_time3, end_discharge_time4],[SCALE_FACTOR1_DISHCARGE,SCALE_FUCTOR2_DISCHARGE,SCALE_FUCTOR3_DISCHARGE,SCALE_FUCTOR4_DISCHARGE])
    #st.write(merged_df1,merged_df)
    final_profile=merged_df.add(merged_df1)
    total_charge,per_car_charge_list=calculateTotalEnergy_EV_Charge(merged_df)
    total_discharge,per_car_discharge_list=calculateTotalEnergy_EV_Charge(merged_df1)
    with st.container():
        fig, ax = plt.subplots(figsize=(10, 4))
        # Create a line plot
        sns.lineplot(data=final_profile[['EV1_charge (W)', 'EV2_charge (W)', 'EV3_charge (W)', 'EV4_charge (W)', 'Total_EV_Charge (W)','TotalImbalance','Imbalance (W)']])
        # Set plot title and axis labels
        plt.title('Electric Vehicle Charging')
        plt.xlabel('Time')
        plt.ylabel('Charge (W)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    with st.container():
        col1,col2,col3=st.columns([2,1,0.9])
        with col1:
            fig1, ax1 = plt.subplots()
            plt.bar(range(len(per_car_charge_list)), per_car_charge_list, color='b')
            # plot bar chart with discharge values
            plt.bar(range(len(per_car_discharge_list)), per_car_discharge_list, color='r')
            # set x-axis labels
            plt.xticks(range(len(per_car_charge_list)), ['Car 1', 'Car 2', 'Car 3', 'Car 4'])
            # set y-axis label
            plt.ylabel('Energy (Wh)')
            # set title
            plt.title('Energy Profile for EVs')
            # show plot
            st.pyplot(fig1)
        with col2:
            fig2, ax2= plt.subplots(figsize=(4, 7))
            plt.bar(['Total Discharge', 'Total Charge'], [total_discharge, total_charge], color=['r', 'b'])
            plt.ylabel('Energy (Wh)')
            plt.title('Total Energy')
            st.pyplot(fig2)
            good_energy_count,bad_energy_count,total_EV_demand_count,per_car_count_list=count_positive_charge_negative_imbalance(merged_df)
        with col3:
            st.pyplot(plot_pie_chart(["PVs","From Grid"],[good_energy_count,bad_energy_count]))
            text1,text2=st.columns([2,1])
            with text1:
                st.markdown(f"Green Energy Used For Charging: {str(good_energy_count)} Wh :👍:")
                st.markdown(f"Red Energy Used For Charging: {str(bad_energy_count)} Wh :👎:")
                st.markdown(f"Total Energy Used For Charging: {str(bad_energy_count+good_energy_count)} Wh :🚙:")
            with text2:
                cars = {'Car Number': ['EV1','EV2','EV3','EV4'], 'Total Energy Used (Wh)': per_car_count_list}
                carts = pd.DataFrame(cars)
                carts=carts.set_index('Car Number')
                st.table(carts)
if __name__ == '__main__':
     sns.set(style="darkgrid")
     DAY=st.selectbox('Select Day',range(1,322))
     main()