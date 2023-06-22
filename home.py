import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")

def seasonal_pv_analysis(data):
    # Convert the "Time" column to datetime
    data['Time'] = pd.to_datetime(data['Time'])
    
    # Create a new column for seasons
    data['Season'] = data['Time'].dt.month.map({1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                                                6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn',
                                                11: 'Autumn', 12: 'Winter'})
    
    # Group the data by season and calculate the total PV generation
    seasonal_pv = data.groupby('Season')['PV (W)'].sum()
    
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the PV generation over time
    plt.ylabel('PV generation (kWh)')
    plt.bar(seasonal_pv.index, -seasonal_pv*0.0025, color='orange')
    
    # Display the figure
    return fig

def seasonal_demand_analysis(data):
    # Convert the "Time" column to datetime
    data['Time'] = pd.to_datetime(data['Time'])
    
    # Create a new column for seasons
    data['Season'] = data['Time'].dt.month.map({1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                                                6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn',
                                                11: 'Autumn', 12: 'Winter'})
    
    # Calculate the weekly demand
    data['Demand'] = data['General Demand (W)'] + data['Heating Demand (W)']
    
    # Group the data by season and calculate the total demand
    seasonal_demand = data.groupby('Season')['Demand'].sum()
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the demand by season
    plt.ylabel('Demand (kwH)')
    plt.bar(seasonal_demand.index, seasonal_demand*0.0025, color='blue')
    
    # Display the figure
    plt.show()
    
    return fig

def average_demand_by_day(data):
    # Convert the "Time" column to datetime
    data['Time'] = pd.to_datetime(data['Time'])
    
    # Extract the day of the week
    data['DayOfWeek'] = data['Time'].dt.dayofweek.map({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                                                       4: 'Friday', 5: 'Saturday', 6: 'Sunday'})
    
    # Calculate the average demand for each day of the week
    data['Demand'] = data['General Demand (W)'] + data['Heating Demand (W)']
    average_demand = data.groupby('DayOfWeek')['Demand'].mean()
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the average demand by day of the week
    plt.ylabel('Average Demand (kwH)')
    plt.bar(average_demand.index, average_demand*0.0025, color='blue')
    
    # Display the figure
    plt.show()
    
    return fig
def main():
    data=pd.read_csv('data_original.csv')
    charging_image = Image.open('imgs/frontCover.png')
    aardenhuizen_image=Image.open('imgs/image.png')
    st.markdown('## _Welcome to the Tool to Investigate the impact of 4 Evs at Aardenhuizen region_')
    c1,c2 = st.columns([3,2])
    with st.container():
        c11,c12 = st.columns([0.7,1])
        with c11:
            st.image(aardenhuizen_image,width=500)
        with c12:
             st.markdown("""The Aardenhuizen community is a small region consisting of 24 households .</br>
                        The community has installed 327 solar panels to generate electricity and 4 electric vehicles (EVs) to promote sustainable transportation.</br>
                        The community members are committed to reducing their carbon footprint by adopting environmentally friendly practices. </br>
                        They prioritize the use of renewable energy sources, such as solar power, to minimize their reliance on non-renewable energy sources.</br>""",unsafe_allow_html=True)
        
    c1,c2,c3 = st.columns([1,1,1])
    with c1:
        seasonal_pv = seasonal_pv_analysis(data)
        st.subheader('Seasonal PV generation')
        st.pyplot(seasonal_pv)
    with c2:
        st.subheader('Seasonal Demand')
        st.pyplot(seasonal_demand_analysis(data))
    with c3:
        st.subheader('Average Demand by Day')
        st.pyplot(average_demand_by_day(data))
            
        
   
    st.markdown('## _Check the DailyEVPlanner on the left sidebar_')


if __name__ == '__main__':
    main()
