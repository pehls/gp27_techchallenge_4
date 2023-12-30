import pandas as pd
import numpy as np
import config
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from datetime import datetime

@st.cache_data
def _df_petroleo():
    df = pd.read_html('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view', decimal=',', thousands='.', parse_dates=True)[2][1:]
    df.columns=['Date','Preco']
    df.Date = pd.to_datetime(df.Date, dayfirst=True)
    df = df.sort_values('Date')
    return df

@st.cache_data
def _series_for_seasonal():
    df = _df_petroleo()
    series = df['Preco']
    series.index = pd.to_datetime(df['Date'])
    return series

@st.cache_data
def _adfuller(series):
    return adfuller(series)

@st.cache_data
def _get_modelling_data(df = _df_petroleo()):
    return df

@st.cache_data
def _get_data_for_models_ts():
    df = _get_modelling_data()
    df_ts = pd.DataFrame(df[['Date', 'Preco']].values, columns=['ds', 'y'])
    df_ts['unique_id'] = 'PETR4'
    df_ts['ds'] = pd.to_datetime(df_ts['ds'])
    df_ts.sort_values('ds', inplace=True)

    date_limit = '2023-01-01'

    train = df_ts.loc[df_ts['ds'] < date_limit]
    test = df_ts.loc[df_ts['ds'] >= date_limit]
    h = test.index.nunique()

    return train, test, h

@st.cache_data
def _trials():
    return pd\
        .read_csv(f'{config.BASE_PATH}/raw/trials.csv')

@st.cache_data
def _events_per_country():
    df_conflitos = pd.read_csv('./data/raw/df_conflitos_porpais.csv')
    df_conflitos['Date'] = [datetime.fromtimestamp(x).date() for x in df_conflitos.timestamp]
    return df_conflitos\
        .groupby(['Date','event_type','country'])\
        .agg({
            'fatalities':'sum'
            , 'event_id_cnty':'nunique'
        }).reset_index()
    

@st.cache_data
def _events_globally():
    df_conflitos = pd.read_csv('./data/raw/df_conflitos_mundiais.csv')
    df_conflitos['Date'] = [datetime.fromtimestamp(x).date() for x in df_conflitos.timestamp]
    return df_conflitos\
        .groupby(['Date','event_type'])\
        .agg({
            'fatalities':'sum'
            , 'event_id_cnty':'nunique'
        }).reset_index()

@st.cache_data
def _type_subtype():
    df_conflitos = pd.read_csv('../data/raw/tipo_subtipo_eventos.csv')
    return df_conflitos[['event_type', 'sub_event_type']].drop_duplicates()