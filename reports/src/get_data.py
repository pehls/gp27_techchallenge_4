import pandas as pd
import numpy as np
import config
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

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
    return df.rename(columns={'Date':'ds', 'Preco':'y'})

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
    df_conflitos['Date'] = pd.to_datetime(df_conflitos['Date'])
    return df_conflitos\
        .groupby(['Date','event_type','country'])\
        .agg({
            'fatalities':'sum'
            , 'event_id_cnty':'sum'
        }).reset_index()
    

@st.cache_data
def _events_globally():
    df_conflitos = pd.read_csv('./data/raw/df_conflitos_mundiais.csv')
    df_conflitos['Date'] = pd.to_datetime(df_conflitos['Date'])
    return df_conflitos\
        .groupby(['Date','event_type'])\
        .agg({
            'fatalities':'sum'
            , 'event_id_cnty':'sum'
        }).reset_index()

@st.cache_data
def _type_subtype():
    df_conflitos = pd.read_csv('../data/raw/tipo_subtipo_eventos.csv')
    return df_conflitos[['event_type', 'sub_event_type']].drop_duplicates()

@st.cache_data
def _events_normalized_globally():
    # ajustando dados de eventos
    df_conflitos_mundiais = _events_globally()
    df_conflitos_mundiais['Date'] = pd.to_datetime(df_conflitos_mundiais['Date'])
    # return df_conflitos_mundiais
    df_conflitos_mundiais = df_conflitos_mundiais.pivot(index='Date', columns='event_type').reset_index().replace(0, None)
    df_conflitos_mundiais.columns = ['Date','Fatalities in Battles','Fatalities in Explosions/Remote Violence','Fatalities in Violence against civillians', 'Qtt Battles','Qtt Explosions/Remote violence','Qtt Violence against civilians']
    # PETR4 price
    df = _df_petroleo()
    df = df.loc[df.Date.isin(df_conflitos_mundiais.Date.to_list())]
    df_conflitos_mundiais = df_conflitos_mundiais[list(df_conflitos_mundiais['Date'] <= pd.to_datetime('2023-12-01'))]
    df_conflitos_mundiais = df_conflitos_mundiais.merge(df, how='inner', on=['Date'])
    for col in df_conflitos_mundiais.columns:
        if (col != 'Date'):
            df_conflitos_mundiais[col] = df_conflitos_mundiais[col].astype(float)
    df_conflitos_mundiais['dt_year_month'] = [x.replace(day=1) for x in df_conflitos_mundiais.Date]
    df_grouped_conf_mund = df_conflitos_mundiais.groupby('dt_year_month').agg({
        'Fatalities in Battles':'sum'
        ,'Fatalities in Explosions/Remote Violence':'sum'
        ,'Fatalities in Violence against civillians':'sum'
        ,'Qtt Battles':'sum'
        ,'Qtt Explosions/Remote violence':'sum'
        ,'Qtt Violence against civilians':'sum'
        ,'Preco':'mean'
    }).reset_index()
    df_grouped_conf_mund['Total Fatalities'] = df_grouped_conf_mund['Fatalities in Battles'] + df_grouped_conf_mund['Fatalities in Explosions/Remote Violence'] + df_grouped_conf_mund['Fatalities in Violence against civillians']
    df_grouped_conf_mund['Total Qtt'] = df_grouped_conf_mund['Qtt Battles'] + df_grouped_conf_mund['Qtt Explosions/Remote violence'] + df_grouped_conf_mund['Qtt Violence against civilians']
    cols_to_reescale = ['Fatalities in Battles','Fatalities in Explosions/Remote Violence','Fatalities in Violence against civillians', 'Qtt Battles','Qtt Explosions/Remote violence','Qtt Violence against civilians', 'Preco','Total Qtt','Total Fatalities']
    scaler = MinMaxScaler()
    df_grouped_conf_mund[["minmax_"+x for x in cols_to_reescale]] = scaler.fit_transform(df_grouped_conf_mund[cols_to_reescale])
    return df_grouped_conf_mund.rename(columns={'dt_year_month':'Date'})

@st.cache_data
def _events_correlations(df_conflitos_preco_normalizados):
    # ajustando dados de eventos
    df_correlacoes = df_conflitos_preco_normalizados.corr()
    cols_to_plot = [x for x in df_correlacoes.columns if x.startswith('minmax_')]
    df_correlacoes = df_correlacoes.loc[cols_to_plot][cols_to_plot]
    df_correlacoes.index = [x.replace('minmax_','') for x in df_correlacoes.index]
    df_correlacoes.columns = [x.replace('minmax_','') for x in df_correlacoes.columns]
    return df_correlacoes