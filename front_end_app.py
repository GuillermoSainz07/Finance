import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import yfinance as yf
from class_model_prediction import Vol_Pred

st.set_page_config(page_title='VolPredictor',
                   layout="wide")

if 'page' not in st.session_state: st.session_state.page = 0

def nextPage():
     st.session_state.page = 1

ph = st.empty()


if st.session_state.page == 0:
    with ph.container():

        st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; height: 200px;">
                <h1 style="text-align: center;">Volatility Predictor</h1>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)


        with col2:
            option = st.text_input('**Stock Ticker**',
                                placeholder='Type a ticker',
                                key='dis')
        
            option = option.capitalize()


        with col1:

            selected_model = st.radio(
                '**Models available:**',
                key='available',
                options=['ARCH','GARCH',
                        'EGARCH','XGBoost',
                        'RandomForest'])
    
# Creacion del boton   
        if len(st.session_state.available) > 0  and len(st.session_state.dis) > 0:
            st.button('ForecastTime!',
                    type='primary',
                    disabled=False,
                    on_click=nextPage)
        
        else:
            st.button('ForecastTime!',
                    type='primary',
                    key='button',
                    disabled=True,
                    on_click=nextPage)

elif st.session_state.page == 1:
    with ph.container():

        st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; height: 200px;">
                <h1 style="text-align: center;">Results of volatility forecast to stock returns</h1>
            </div>
        """, unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            st.header('Plots and errors')
            construct = Vol_Pred(st.session_state.dis,
                             start='2018-01-01')
            if st.session_state.available == 'xgb':
                performance_plots, df_fold = construct.create_model(type_model='xgb',
                                                                performance=True)
            else:
                performance_plots, df_fold = construct.create_model(type_model='randomforest',
                                                                performance=True)
            st.plotly_chart(performance_plots)
            st.write(df_fold)
        with col4:
            st.header('Prediction to the future and feature importance')
     
            prediction, faeture_importance = construct.predict_future()

            st.write(prediction)
            st.write(faeture_importance)
        
        


