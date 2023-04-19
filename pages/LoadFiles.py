import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

st. set_page_config(layout="wide")
# Use the SessionState class to store the uploaded files
session_state = st.session_state
if 'file1' not in session_state:
    session_state.file1 = None
if 'file2' not in session_state:
    session_state.file2 = None

file1 = st.file_uploader("Upload Strategy1", type=['csv', 'xlsx'])
if file1 is not None:
    session_state.file1 = pd.read_csv(file1)  # or pd.read_excel(file1) for XLSX files
    try:
        session_state.file1.drop(columns=['level_0','index','Hour','DayOfWeek'], inplace=True)
    except:
        pass
    session_state.file1.fillna(0, inplace=True)
    
file2 = st.file_uploader("Upload Strategy2", type=['csv', 'xlsx'])
if file2 is not None:
    session_state.file2 = pd.read_csv(file2)  # or pd.read_excel(file2) for XLSX files
    try:
        session_state.file2.drop(columns=['level_0','index','Hour','DayOfWeek'], inplace=True)
    except:
        pass
    session_state.file2.fillna(0, inplace=True)

# Use the uploaded files as session states in other parts of your code
col1,col2=st.columns([2,2])
with col1:
    if session_state.file1 is not None:
        #st.write(session_state.file1)
        #st.write(session_state.file1.describe())
        
        df1=session_state.file1
        df2=session_state.file2
        fig , ax = plt.subplots()
        ax.plot(df1.index,df1['Imbalnace'].rolling(4*7*96).mean())
        st.pyplot(fig)
        
with col2:
    if session_state.file2 is not None:
        #st.write(session_state.file2)
        #st.write(session_state.file2.describe())
        fig1 , ax1 = plt.subplots()
        ax1.plot(df2.index,df2['Imbalnace'].rolling(4*7*96).mean())
        st.pyplot(fig1)