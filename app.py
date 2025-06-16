import streamlit as st

# Display loading message
# st.write("Loading NMI Consumption Forecast Dashboard...")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from joblib import dump
from dynaconf import settings
import pyodbc
import logging
# from joblib import Memory
from train_and_pred import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from openai import OpenAI

# Set page config
st.set_page_config(
    page_title="NMI Consumption Forecast Dashboard",
    layout="wide"
)

# Initialize session state for caching
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'custom_facts' not in st.session_state:
    st.session_state.custom_facts = []
if 'previous_mape' not in st.session_state:
    st.session_state.previous_mape = None
if 'previous_nmi' not in st.session_state:
    st.session_state.previous_nmi = None
if 'previous_forecast_dates' not in st.session_state:
    st.session_state.previous_forecast_dates = None
if 'previous_forecast_values' not in st.session_state:
    st.session_state.previous_forecast_values = None

def load_data(nmi_id, start_date, end_date, facts=[], test_dates=[]):
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Load and process data with progress updates
    with st.spinner('Loading historical consumption data...'):
        df_nmi_X, df_nmi_Y = get_train_test_data(nmi_id, start_date, end_date)
        progress_bar.progress(20)
    
    with st.spinner('Splitting data into training and testing sets...'):
        df_nmi_test_X, df_nmi_train_X, df_nmi_test_y, df_nmi_train_y = train_test_split(df_nmi_X, df_nmi_Y, test_dates)
        progress_bar.progress(40)
    
    with st.spinner('Integrating domain knowledge and processing features...'):
        client = OpenAI(api_key=st.secrets['default']["OPENAI_API_KEY"])
        df_nmi_train_X, df_nmi_test_X, df_future, df_LLM, codes = integrate_domain_knowledge(client, nmi_id, start_date, end_date, df_nmi_train_X, df_nmi_test_X, facts)
        progress_bar.progress(60)
    
    with st.spinner('Preparing final training and testing datasets...'):
        df_nmi_train_X, df_nmi_test_X, df_nmi_train_y, df_nmi_test_y = prep_train_test_data(df_nmi_train_X, df_nmi_test_X, df_nmi_train_y, df_nmi_test_y)
        progress_bar.progress(70)
    
    with st.spinner('Training the forecasting model...'):
        model, rmse_train, mape_train = train_model(df_nmi_train_X, df_nmi_train_y, nmi_id)
        progress_bar.progress(80)
    
    with st.spinner('Evaluating model performance...'):
        df_nmi_test_y_pred, rmse_test, mape_test = predict_model(model, df_nmi_test_X, df_nmi_test_y)   
        progress_bar.progress(90)
    
    with st.spinner('Generating future consumption forecasts...'):
        df_future_dates, y_pred = forecast_nmi_consumption(model, nmi_id, start_date, end_date, df_future, codes)
        progress_bar.progress(100)
    
    return model, mape_test, df_nmi_X, df_nmi_Y, df_future_dates, y_pred, df_LLM, df_future

# def get_all_meter_details():
#     df_meter_info = get_meter_info()
#     meter_dict = {row['NetworkID'] : row['Name'] for _, row in df_meter_info.iterrows()}
#     campus_dict = {row['NetworkID'] : row['CampusKey'] for _, row in df_meter_info.iterrows()}
#     return meter_dict, campus_dict

# Load configuration data
@st.cache_data
def load_config_data():
    meter_dict, campus_dict = get_all_meter_details()
    with open('metadata/facts.json', 'r') as f:
        facts_data = json.load(f)
    with open('metadata/train_test_split.json', 'r') as f:
        test_split_data = json.load(f)
    return meter_dict, campus_dict, facts_data, test_split_data

# Main app
def main():
    st.title("NMI Consumption Forecast Dashboard")
    
    # Load configuration data
    meter_dict, campus_dict, facts_data, test_split_data = load_config_data()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    USE_FACTS = st.sidebar.checkbox("Use Domain Facts", value=True)
    
    # Select NMI
    nmi_id = st.sidebar.selectbox(
        "Select NMI",
        options=list(meter_dict.keys()),
        format_func=lambda x: f"{x} - {meter_dict[x]}"
    )
    
    # Reset previous values if NMI selection changes
    if st.session_state.previous_nmi != nmi_id:
        st.session_state.previous_mape = None
        st.session_state.previous_forecast_dates = None
        st.session_state.previous_forecast_values = None
        st.session_state.previous_nmi = nmi_id
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2018, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime(2025, 6, 1))
    
    # Convert dates to required format
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    # Get facts and test split data
    suggested_facts = facts_data.get(str(nmi_id), [])
    test_split = test_split_data.get(str(nmi_id), {})
    
    # Display suggested facts
    st.sidebar.subheader("Suggested Domain Facts")
    for fact in suggested_facts:
        st.sidebar.info(fact)
    
    # Add custom facts input
    st.sidebar.subheader("Add Domain Facts")
    new_fact = st.sidebar.text_area("Enter your domain fact here", height=100)
    if st.sidebar.button("Add Fact"):
        if new_fact:
            st.session_state.custom_facts.append(new_fact)
            new_fact = ""
    
    # Display custom facts
    if st.session_state.custom_facts:
        st.sidebar.subheader("Your Added Facts")
        for i, fact in enumerate(st.session_state.custom_facts):
            st.sidebar.info(fact)
            if st.sidebar.button(f"Delete Fact {i+1}", key=f"delete_{i}"):
                st.session_state.custom_facts.pop(i)
                st.rerun()
    
    # Load and process data
    if st.sidebar.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            if USE_FACTS:
                model, mape_test, df_nmi_X, df_nmi_Y, df_future_dates, y_pred, df_LLM, df_LLM_future = load_data(
                    nmi_id, start_date_str, end_date_str, facts=st.session_state.custom_facts, test_dates=test_split
                )
            else:
                model, mape_test, df_nmi_X, df_nmi_Y, df_future_dates, y_pred, _, _ = load_data(
                    nmi_id, start_date_str, end_date_str, test_dates=test_split
                )
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add traces
            # Commented out gross consumption for now
            # fig.add_trace(go.Scatter(
            #     x=df_nmi_Y['DatePlot'],
            #     y=df_nmi_Y['consumption'],
            #     name='NMI Consumption (Gross)',
            #     line=dict(color='red')
            # ))
            
            fig.add_trace(go.Scatter(
                x=df_nmi_Y['DatePlot'],
                y=df_nmi_Y['metered_consumption'],
                name='Historical Consumption',
                line=dict(color='blue')
            ))
            
            # Commented out gross prediction for now
            # fig.add_trace(go.Scatter(
            #     x=df_future_dates['DatePlot'],
            #     y=y_pred,
            #     name='Predicted NMI Consumption (Gross)',
            #     line=dict(color='rgba(191, 110, 97, 1)')
            # ))
            
            # Add previous forecast if available
            if st.session_state.previous_forecast_dates is not None:
                fig.add_trace(go.Scatter(
                    x=st.session_state.previous_forecast_dates,
                    y=st.session_state.previous_forecast_values,
                    name='Previous Forecast',
                    line=dict(color='rgba(169, 169, 169, 0.7)')  # grey
                ))

            # Add current forecast
            fig.add_trace(go.Scatter(
                x=df_future_dates['DatePlot'],
                y=y_pred,
                name='Forecast',
                line=dict(color='rgb(34, 139, 34)')  # forest green
            ))

            # Store current forecast for next run
            st.session_state.previous_forecast_dates = df_future_dates['DatePlot']
            st.session_state.previous_forecast_values = y_pred
            
            # Update layout
            fig.update_layout(
                title=f"NMI Consumption Forecast - {meter_dict[nmi_id]} (Model Error (↓ the better): {mape_test:.2%})",
                xaxis_title="Date",
                yaxis_title="kWh",
                showlegend=True,
                height=600,
                template='plotly_white'
            )
            
            # Add grid
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
                        # Display model error metric above the plot
            if st.session_state.previous_mape is not None:
                mape_delta = mape_test - st.session_state.previous_mape
                st.metric(
                    "Model Error (↓ the better)", 
                    f"{mape_test:.2%}",
                    f"{mape_delta:.2%}",
                    delta_color="inverse"
                )
            else:
                st.metric("Model Error (↓ the better)", f"{mape_test:.2%}")
            
            # Store current MAPE for next comparison
            st.session_state.previous_mape = mape_test
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display additional information in columns
            col1, col2 = st.columns(2)
            
            # Display facts and test period information in an expander
            with st.expander("Additional Information", expanded=False):
                if USE_FACTS and st.session_state.custom_facts:
                    st.subheader("Domain Facts")
                    for fact in st.session_state.custom_facts:
                        st.info(fact)
                
                if test_split:
                    st.subheader("Test Period")
                    test_start = str(test_split.get('test_start_date', ''))
                    test_end = str(test_split.get('test_end_date', ''))
                    if test_start and test_end:
                        test_start_formatted = f"{test_start[:4]}-{test_start[4:6]}-{test_start[6:]}"
                        test_end_formatted = f"{test_end[:4]}-{test_end[4:6]}-{test_end[6:]}"
                        st.write(f"{test_start_formatted} to {test_end_formatted}")

if __name__ == "__main__":
    main()
