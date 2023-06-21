import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st  

def plot_imbalance_effect(uploaded_file):
    # Load the data into a Pandas DataFrame
    #df = pd.read_csv(uploaded_file)
    df['TotalImbalance'] = df['TotalImbalance'] + df['Total_EV_DisCharge (W)']

    # Calculate the average positive and negative imbalances for 'Imbalance (W)'
    average_positive_imbalance = df[df['Imbalance (W)'] > 0]['Imbalance (W)'].mean()
    average_negative_imbalance = df[df['Imbalance (W)'] < 0]['Imbalance (W)'].mean()

    # Calculate the average positive and negative imbalances for 'TotalImbalance'
    average_positive_imbalance_2 = df[df['TotalImbalance'] > 0]['TotalImbalance'].mean()
    average_negative_imbalance_2 = df[df['TotalImbalance'] < 0]['TotalImbalance'].mean()

    # Create the line plot
    fig = plt.figure(figsize=(15, 6))
    plt.plot([average_positive_imbalance] * len(df), color='red', linestyle='dotted', label=f'Average Positive Imbalance: {round(average_positive_imbalance)}')
    plt.plot([average_negative_imbalance] * len(df), color='red', linestyle='--', label=f'Average Negative Imbalance: {round(average_negative_imbalance)}')
    plt.plot([average_positive_imbalance_2] * len(df), color='blue', linestyle='dotted', label=f'Average Positive Imbalance with 4 EVs: {round(average_positive_imbalance_2)}')
    plt.plot([average_negative_imbalance_2] * len(df), color='blue', linestyle='--', label=f'Average Negative Imbalance with 4 EVs: {round(average_negative_imbalance_2)}')
    imbalance_values = df['Imbalance (W)']
    imbalance_values_2 = df['TotalImbalance']
    plt.plot(imbalance_values, label='Imbalance (W)', color='red')
    plt.plot(imbalance_values_2, label='Imbalance With 4 EVs (W)', color='blue')
    plt.axhline(0, color='black')
    plt.xlabel('Time')
    plt.ylabel('Imbalance (W)')
    plt.title('Impact of 4 EVs on the Imbalance Curve')
    plt.legend()
    plt.show()

    effect_on_positive_imbalance = ((average_positive_imbalance - average_positive_imbalance_2) / average_positive_imbalance) * 100
    effect_on_negative_imbalance = ((average_negative_imbalance - average_negative_imbalance_2) / average_negative_imbalance) * 100
    plt.savefig('Imbalance.png')
    return fig

def create_total_demand_vs_ev_charge_plot(upload_file):
    #df=pd.read_csv(upload_file)
    total_demand = df['TotalDemand (W)'].sum()
    total_ev_charge = df['Total_EV_Charge (W)'].sum()
    total_ev_discharge = df['Total_EV_DisCharge (W)'].sum()
    hourly_total_demand = df['TotalDemand (W)'].tail(4).sum()
    total_demand_sum = df.iloc[76:85]['TotalDemand (W)'].sum()
    effect = ((total_demand_sum) / total_ev_discharge) * 100

    data = {'Category': [ 'Total_EV_Discharge', '2-hours-Peak Demand'],
            'Value': [ -total_ev_discharge*0.0025, total_demand_sum*0.0025]}
    df_plot = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(6,12))
    ax.bar(df_plot['Category'], df_plot['Value'])

    ax.set_ylabel('Value (kWh)')
    ax.set_title('Total Demand vs Total EV Charge')

    return fig,effect
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

def plot_pie_chart(labels, values):
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%',colors=('g','r'))
    
    ax.set_aspect('equal')
    
    return fig


def main(data):
    c1,c2=st.columns([2,1])
    with c1:
        fig1 = plot_imbalance_effect(data)
        st.pyplot(fig1)
        car1_init_charge = data['BatteryLVL1'].head(1)
        car2_init_charge = data['BatteryLVL2'].head(1)
        car3_init_charge = data['BatteryLVL3'].head(1)
        car4_init_charge = data['BatteryLVL4'].head(1)
        data1=calculate_energy_storage(data,4,[car1_init_charge,car2_init_charge,car3_init_charge,car4_init_charge])
        c11,c12=st.columns(2)
        with c11:
            st.subheader("Battery Levels")
            car1_battery_lvl1 = data1['BatteryLVL1'].tail(1)
            car2_battery_lvl1 = data1['BatteryLVL2'].tail(1)
            car3_battery_lvl1 = data1['BatteryLVL3'].tail(1)
            car4_battery_lvl1 = data1['BatteryLVL4'].tail(1)
            st.write(':battery:_1 = '+str(round(car1_battery_lvl1.values[0]/300000,2)*100)+'%')
            st.write(':battery:_2 = '+str(round(car2_battery_lvl1.values[0]/300000,2)*100)+'%')
            st.write(':battery:_3 = '+str(round(car3_battery_lvl1.values[0]/300000,2)*100)+'%')
            st.write(':battery:_4 = '+str(round(car4_battery_lvl1.values[0]/300000,2)*100)+'%')
        with c12:
            st.subheader("Green Energy Penetration")
            good_energy_count,bad_energy_count,total_EV_demand_count,per_car_count_list=count_positive_charge_negative_imbalance(data)
            st.pyplot(plot_pie_chart(["PVs","From Grid"],[good_energy_count,bad_energy_count]))
    with c2:
        fig2,effect = create_total_demand_vs_ev_charge_plot(data)
        st.pyplot(fig2)
        
        
        
        
        
    

if __name__ == "__main__":
    data=st.file_uploader("Upload a file", type="csv")
    df=pd.read_csv(data)
    main(df)
    
    

