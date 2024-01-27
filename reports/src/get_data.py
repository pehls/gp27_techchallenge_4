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
def _events_correlations(df_conflitos_preco_normalizados, cols_to_plot=None):
    # ajustando dados de eventos
    df_correlacoes = df_conflitos_preco_normalizados.corr()
    if not(cols_to_plot):
        cols_to_plot = [x for x in df_correlacoes.columns if x.startswith('minmax_')]
    df_correlacoes = df_correlacoes.loc[cols_to_plot][cols_to_plot]
    df_correlacoes.index = [x.replace('minmax_','') for x in df_correlacoes.index]
    df_correlacoes.columns = [x.replace('minmax_','') for x in df_correlacoes.columns]
    return df_correlacoes

@st.cache_data
def _df_energy_use_top10():
    df_uso_energia_or = pd.read_csv('data/raw/energy_use/API_EG.USE.PCAP.KG.OE_DS2_en_csv_v2_6301176.csv').drop(columns={'Indicator Name','Indicator Code'})
    df_country_region = pd.read_csv('data/raw/energy_use/Metadata_Country_API_EG.USE.PCAP.KG.OE_DS2_en_csv_v2_6301176.csv')[['Country Code','Region']].dropna()
    df_uso_energia_or = df_uso_energia_or.merge(df_country_region, how='inner', on='Country Code').drop(columns={'Country Code','Region'})
    cols_to_plot = [x for x in list(set(df_uso_energia_or.columns)-set(['Country Name']))]
    cols_to_plot.sort(reverse=True)
    df_uso_energia_or = df_uso_energia_or.dropna(axis=1, thresh=0.95)

    cols_to_plot = [x for x in list(set(df_uso_energia_or.columns)-set(['Country Name']))]
    cols_to_plot.sort(reverse=True)
    cols_to_plot = cols_to_plot[:5]
    df_uso_energia_or = df_uso_energia_or[['Country Name'] + cols_to_plot]
    df_uso_energia_or['Total En. Use 11-15'] = [round(x, 2) for x in df_uso_energia_or[cols_to_plot].sum(axis=1)]
    df_uso_energia_or['mean'] = [round(x, 2) for x in df_uso_energia_or[cols_to_plot].mean(axis=1)]
    df_uso_energia_or = df_uso_energia_or\
        .sort_values('Total En. Use 11-15', ascending=False)\
        .head(10)
    df_uso_energia_or['2015'] = np.where(np.isnan(df_uso_energia_or['2015']), df_uso_energia_or['mean'], df_uso_energia_or['2015'])
    return df_uso_energia_or.sort_values('Total En. Use 11-15')

@st.cache_data
def _df_energy_use(lista_paises = None, full=False):
    # selecionar colunas numericas e nome
    cols = [str(x) for x in ['Country Name'] + list(range(1960, 2023))]
    df_uso_energia = pd.read_csv('data/raw/energy_use/API_EG.USE.PCAP.KG.OE_DS2_en_csv_v2_6301176.csv')
    df_uso_energia = df_uso_energia.dropna(axis=1, thresh=0.9)
    if (full):
        df_country_region = pd.read_csv('data/raw/energy_use/Metadata_Country_API_EG.USE.PCAP.KG.OE_DS2_en_csv_v2_6301176.csv')[['Country Code','Region']].dropna()
        df_uso_energia = df_uso_energia.merge(df_country_region, how='inner', on='Country Code').drop(columns={'Country Code','Region'})
        df_uso_energia = df_uso_energia.dropna(axis=1, thresh=0.95)
        return df_uso_energia

    df_uso_energia = df_uso_energia.loc[df_uso_energia['Country Name'].isin(lista_paises)]

    # ajustar df de petroleo
    df_petroleo = _df_petroleo()
    df_petroleo['Year'] = [str(x.year) for x in df_petroleo['Date']]
    df_petroleo['Preco'] = df_petroleo['Preco'].astype(float) 
    df_petroleo_year = df_petroleo.groupby('Year').agg({'Preco':'mean'}).reset_index()
    df_petroleo_year['Country Name'] = 'Price'

    # colunas em comum
    cols = list(set(df_uso_energia.columns).intersection(df_petroleo_year.Year.values) - set(['Country Name']))
    df_petroleo_year = df_petroleo_year.loc[df_petroleo_year.Year.isin(cols)]

    # normalizar apenas o preço que estará no comparativo
    scaler_py = MinMaxScaler()
    df_petroleo_year[["value"]] = scaler_py.fit_transform(df_petroleo_year[['Preco']])
    # pivotar para unir com df_uso_energia
    df_petroleo_year = df_petroleo_year[['Country Name','Year','value']]
    df_petroleo_year.columns = [str(x) for x in df_petroleo_year.columns]

    # colunas em comum
    df_uso_energia = df_uso_energia[['Country Name'] + cols]
    
    # vamos preencher o na de 2015 de alguns paises com a media dos anos de 11-15
    df_uso_energia['mean'] = [round(x, 2) for x in df_uso_energia[['2011','2012','2013','2014','2015']].mean(axis=1)]
    df_uso_energia['2015_nulo'] = np.where(np.isnan(df_uso_energia['2015']), 1, 0)
    df_uso_energia['2015']  = np.where(np.isnan(df_uso_energia['2015']), df_uso_energia['mean'], df_uso_energia['2015'])
    lista_2015_nulo = [f'-{x}' for x in df_uso_energia.loc[df_uso_energia['2015_nulo']==1]['Country Name'].values]
    df_uso_energia = df_uso_energia.drop(columns={'2015_nulo','mean'})

    # melt para termos valor e ano, igual ao df de petroleo
    df_uso_energia = df_uso_energia.melt(id_vars='Country Name').rename(columns={'variable':'Year'})
    df_uso_energia = df_uso_energia.groupby(['Country Name', 'Year']).agg({'value':'sum'}).reset_index()
    # reescalar pra comparação
    scaler_en = MinMaxScaler()
    df_uso_energia[["minmax_value"]] = scaler_en.fit_transform(df_uso_energia[['value']])
    
    # pegar apenas colunas de interesse
    cols = ['Country Name','Year','minmax_value']
    df_uso_energia = df_uso_energia[cols]
    cols = [x.replace('minmax_','') for x in cols]
    df_uso_energia.columns = cols
    df_uso_energia = pd.concat([df_uso_energia[cols],df_petroleo_year[cols]])
    return df_uso_energia, lista_2015_nulo

@st.cache_data
def _df_fossil_fuel_cons(full=False):
    #  selecionar colunas numericas e nome
    cols = [str(x) for x in ['Country Name','Region'] + list(range(1960, 2023))]
    df_fuel_cons = pd.read_csv('data/raw/fossil_fuel_consumption/API_EG.USE.COMM.FO.ZS_DS2_en_csv_v2_6299038.csv')
    df_country_region = pd.read_csv('data/raw/fossil_fuel_consumption/Metadata_Country_API_EG.USE.COMM.FO.ZS_DS2_en_csv_v2_6299038.csv')[['Country Code','Region']].dropna()
    df_fuel_cons = df_fuel_cons.merge(df_country_region, how='inner', on='Country Code').drop(columns={'Country Code'})[cols]

    # dropar na
    df_fuel_cons = df_fuel_cons.dropna(axis=1, thresh=0.95)
    if (full):
        return df_fuel_cons
    cols_to_plot = [x for x in list(set(df_fuel_cons.columns)-set(['Country Name','Region']))]
    cols_to_plot.sort(reverse=True)
    cols_to_plot = cols_to_plot[:5]
    df_fuel_cons = df_fuel_cons[['Country Name','Region'] + cols_to_plot[:5]]

    # top10
    df_fuel_cons['Mean Fuel Cons. 11-15'] = [round(x, 2) for x in df_fuel_cons[cols_to_plot].mean(axis=1)]
    df_fuel_cons['Total Fuel Cons. 11-15'] = [round(x, 2) for x in df_fuel_cons[cols_to_plot].sum(axis=1)]
    df_fuel_cons = df_fuel_cons\
        .sort_values('Mean Fuel Cons. 11-15', ascending=False)\
        .head(10)
    df_fuel_cons.index = range(1,11)

    return df_fuel_cons[['Country Name','Region','Mean Fuel Cons. 11-15']]

@st.cache_data
def _get_fossil_fuel_cons_energy_use_corr():
    # ajustar df de petroleo
    df_petroleo = _df_petroleo()
    df_petroleo['Year'] = [str(x.year) for x in df_petroleo['Date']]
    df_petroleo['Preco'] = df_petroleo['Preco'].astype(float) 
    df_petroleo_year = df_petroleo.groupby('Year').agg({'Preco':'mean'}).reset_index()
    df_petroleo_year['Country Name'] = 'Price'

    #fossil
    df_fuel_cons = _df_fossil_fuel_cons(full=True)
    df_fuel_cons = df_fuel_cons\
        .melt(id_vars=['Country Name','Region'])\
        .rename(columns={'variable':'Year', 'value':'Fuel Consumption'})\
        .drop(columns='Region')\
        .pivot(index='Year', columns='Country Name')\
        .reset_index()
    df_fuel_cons.columns = ['_'.join(col) for col in df_fuel_cons.columns.values]
    df_fuel_cons = df_fuel_cons.rename(columns={'Year_':'Year'})
    
    df_uso_energia_or = _df_energy_use(lista_paises = None, full=True)
    df_uso_energia_or = df_uso_energia_or\
        .melt(id_vars=['Country Name'])\
        .rename(columns={'variable':'Year', 'value':'Energy Use'})\
        .pivot(index='Year', columns='Country Name')\
        .reset_index()
    df_uso_energia_or.columns = ['_'.join(col) for col in df_uso_energia_or.columns.values]
    df_uso_energia_or = df_uso_energia_or.rename(columns={'Year_':'Year'})

    return df_fuel_cons\
        .merge(df_petroleo_year[['Year','Preco']], on='Year', how='inner')\
        .merge(df_uso_energia_or, on='Year', how='inner')\
        .replace(0.0, np.nan)\
        .dropna(axis=0, thresh=0.5)\
        .dropna(axis=1, thresh=0.5)

@st.cache_data
def _get_faixas_correlation(df_correlacoes):
    return {
          ' itens com uma correlação menor que 0.3;':len(df_correlacoes.loc[(abs(df_correlacoes[['Preco']]) < 0.3)['Preco']][['Preco']])
        , ' com uma correlação fraca (entre 0.3 e 0.5);':len((df_correlacoes.loc[((abs(df_correlacoes[['Preco']]) >= 0.3) & (abs(df_correlacoes[['Preco']]) < 0.5))['Preco']][['Preco']]))
        , ' com uma correlação moderada (entre 0.5 e 0.7);':len((df_correlacoes.loc[((abs(df_correlacoes[['Preco']]) >= 0.5) & (abs(df_correlacoes[['Preco']]) < 0.7))['Preco']][['Preco']]))
        , ' com uma correlação forte (entre 0.7 e 0.9);':len((df_correlacoes.loc[((abs(df_correlacoes[['Preco']]) >= 0.7) & (abs(df_correlacoes[['Preco']]) < 0.9))['Preco']][['Preco']]))
        , ' com uma correlação muito forte (acima de 0.9);':len((df_correlacoes.loc[((abs(df_correlacoes[['Preco']]) >= 0.9))['Preco']][['Preco']]))-1
    }

@st.cache_data
def _df_fuel_exports(full=False):
    df = pd.read_csv('data/raw/fuel_exports/API_TX.VAL.FUEL.ZS.UN_DS2_en_csv_v2_6302702.csv')
    df_preco = _df_petroleo()
    #  selecionar colunas numericas e nome
    cols = [str(x) for x in ['Country Name','Region'] + list(range(1960, 2023))]
    df_fuel_exp = pd.read_csv('data/raw/fuel_exports/API_TX.VAL.FUEL.ZS.UN_DS2_en_csv_v2_6302702.csv')
    df_country_region = pd.read_csv('data/raw/fuel_exports/Metadata_Country_API_TX.VAL.FUEL.ZS.UN_DS2_en_csv_v2_6302702.csv')[['Country Code','Region']].dropna()
    df_fuel_exp = df_fuel_exp.merge(df_country_region, how='inner', on='Country Code').drop(columns={'Country Code'})[cols]

    # dropar na
    df_fuel_exp = df_fuel_exp.dropna(axis=1, thresh=0.95)
    if full:
        return df_fuel_exp
    cols_to_plot = [x for x in list(set(df_fuel_exp.columns)-set(['Country Name','Region']))]
    cols_to_plot.sort(reverse=True)
    cols_to_plot = cols_to_plot[:5]
    df_fuel_exp = df_fuel_exp[['Country Name','Region'] + cols_to_plot[:5]]

    # top10
    df_fuel_exp['Mean Fuel Exp. 11-15'] = [round(x, 2) for x in df_fuel_exp[cols_to_plot].mean(axis=1)]
    df_fuel_exp['Total Fuel Exp. 11-15'] = [round(x, 2) for x in df_fuel_exp[cols_to_plot].sum(axis=1)]
    df_fuel_exp = df_fuel_exp\
        .sort_values('Mean Fuel Exp. 11-15', ascending=False)\
        .head(10)\
        .reset_index()
    return df_fuel_exp[['Country Name','Region','Mean Fuel Exp. 11-15']]

@st.cache_data
def _df_fuel_exp_corr():
    df = pd.read_html('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view', decimal=',', thousands='.', parse_dates=True)[2][1:]
    df.columns=['Date','Preco']
    df.Date = pd.to_datetime(df.Date, dayfirst=True)
    df_petroleo = df.sort_values('Date')

    # ajustar df de petroleo
    df_petroleo['Year'] = [str(x.year) for x in df_petroleo['Date']]
    df_petroleo['Preco'] = df_petroleo['Preco'].astype(float) 
    df_petroleo_year = df_petroleo.groupby('Year').agg({'Preco':'mean'}).reset_index()
    df_petroleo_year['Country Name'] = 'Price'

    # exportacao de comb
    cols = [str(x) for x in ['Country Name','Region'] + list(range(1960, 2023))]
    df_fuel_exp = pd.read_csv('data/raw/fuel_exports/API_TX.VAL.FUEL.ZS.UN_DS2_en_csv_v2_6302702.csv')
    df_country_region = pd.read_csv('data/raw/fuel_exports/Metadata_Country_API_TX.VAL.FUEL.ZS.UN_DS2_en_csv_v2_6302702.csv')[['Country Code', 'Region']].dropna()
    df_fuel_exp = df_fuel_exp.merge(df_country_region, how='inner', on='Country Code').drop(columns={'Country Code'})[cols]

    # dropar na
    df_fuel_exp = df_fuel_exp.dropna(axis=1, thresh=0.95).drop(columns={'Region'})
    df_fuel_exp = df_fuel_exp\
        .melt(id_vars=['Country Name'])\
        .rename(columns={'variable':'Year', 'value':'Fuel Exports'})\
        .pivot(index='Year', columns='Country Name')\
        .reset_index()
    df_fuel_exp.columns = ['_'.join(col) for col in df_fuel_exp.columns.values]
    df_fuel_exp = df_fuel_exp.rename(columns={'Year_':'Year'})

    df_fuel_exp = df_petroleo_year[['Year','Preco']]\
        .merge(df_fuel_exp, on='Year', how='inner')\
        .replace(0.0, np.nan)\
        .dropna(axis=0, thresh=0.5)\
        .dropna(axis=1, thresh=0.5)
    
    cols_to_reescale = list(set(df_fuel_exp.columns) - set(['Year']))
    scaler = MinMaxScaler()
    df_fuel_exp[["minmax_"+x for x in cols_to_reescale]] = scaler.fit_transform(df_fuel_exp[cols_to_reescale])

    return df_fuel_exp

@st.cache_data
def _df_tree_modelling():
    df = pd.read_html('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view', decimal=',', thousands='.', parse_dates=True)[2][1:]
    df.columns=['Date','Preco']
    df.Date = pd.to_datetime(df.Date, dayfirst=True)
    df_petroleo = df.sort_values('Date')

    # ajustar df de petroleo
    df_petroleo['Year'] = [str(x.year) for x in df_petroleo['Date']]
    df_petroleo['Preco'] = df_petroleo['Preco'].astype(float) 
    df_petroleo_year = df_petroleo.groupby('Year').agg({'Preco':'mean'}).reset_index()
    df_petroleo_year['Country Name'] = 'Price'

    #  consumo de comb
    cols = [str(x) for x in ['Country Name','Region'] + list(range(1960, 2023))]
    df_fuel_cons = pd.read_csv('data/raw/fossil_fuel_consumption/API_EG.USE.COMM.FO.ZS_DS2_en_csv_v2_6299038.csv')
    df_country_region = pd.read_csv('data/raw/fossil_fuel_consumption/Metadata_Country_API_EG.USE.COMM.FO.ZS_DS2_en_csv_v2_6299038.csv')[['Country Code','Region']].dropna()
    df_fuel_cons = df_fuel_cons.merge(df_country_region, how='inner', on='Country Code').drop(columns={'Country Code'})[cols]

    # dropar na
    df_fuel_cons = df_fuel_cons.dropna(axis=1, thresh=0.95)

    df_fuel_cons = df_fuel_cons\
        .melt(id_vars=['Country Name','Region'])\
        .rename(columns={'variable':'Year', 'value':'Fuel Consumption'})\
        .drop(columns='Region')\
        .pivot(index='Year', columns='Country Name')\
        .reset_index()
    df_fuel_cons.columns = ['_'.join(col) for col in df_fuel_cons.columns.values]
    df_fuel_cons = df_fuel_cons.rename(columns={'Year_':'Year'})

    # uso de energia
    df_uso_energia_or = pd.read_csv('data/raw/energy_use/API_EG.USE.PCAP.KG.OE_DS2_en_csv_v2_6301176.csv').drop(columns={'Indicator Name','Indicator Code'})
    df_country_region = pd.read_csv('data/raw/energy_use/Metadata_Country_API_EG.USE.PCAP.KG.OE_DS2_en_csv_v2_6301176.csv')[['Country Code','Region']].dropna()
    df_uso_energia_or = df_uso_energia_or.merge(df_country_region, how='inner', on='Country Code').drop(columns={'Country Code','Region'})
    df_uso_energia_or = df_uso_energia_or.dropna(axis=1, thresh=0.95)
    df_uso_energia_or = df_uso_energia_or\
        .melt(id_vars=['Country Name'])\
        .rename(columns={'variable':'Year', 'value':'Energy Use'})\
        .pivot(index='Year', columns='Country Name')\
        .reset_index()
    df_uso_energia_or.columns = ['_'.join(col) for col in df_uso_energia_or.columns.values]
    df_uso_energia_or = df_uso_energia_or.rename(columns={'Year_':'Year'})

    # exportacao de comb
    cols = [str(x) for x in ['Country Name','Region'] + list(range(1960, 2023))]
    df_fuel_exp = pd.read_csv('data/raw/fuel_exports/API_TX.VAL.FUEL.ZS.UN_DS2_en_csv_v2_6302702.csv')
    df_country_region = pd.read_csv('data/raw/fuel_exports/Metadata_Country_API_TX.VAL.FUEL.ZS.UN_DS2_en_csv_v2_6302702.csv')[['Country Code', 'Region']].dropna()
    df_fuel_exp = df_fuel_exp.merge(df_country_region, how='inner', on='Country Code').drop(columns={'Country Code'})[cols]

    # dropar na
    df_fuel_exp = df_fuel_exp.dropna(axis=1, thresh=0.95).drop(columns={'Region'})
    df_fuel_exp = df_fuel_exp\
        .melt(id_vars=['Country Name'])\
        .rename(columns={'variable':'Year', 'value':'Fuel Exports'})\
        .pivot(index='Year', columns='Country Name')\
        .reset_index()
    df_fuel_exp.columns = ['_'.join(col) for col in df_fuel_exp.columns.values]
    df_fuel_exp = df_fuel_exp.rename(columns={'Year_':'Year'})
    
    # dado de dow jones e nasdaq
    df_dowjones_nasdaq = pd.read_csv('data/df_brent_dowjones_nasdaq_norm.csv')
    df_dowjones_nasdaq['Year'] = [str(pd.to_datetime(x).year) for x in df_dowjones_nasdaq['DATE']]
    df_dowjones_nasdaq = df_dowjones_nasdaq\
        .groupby('Year')\
        .agg({
              'value_dow_jones':('mean', 'median', 'std','min','max')
            , 'value_nasdaq':('mean', 'median', 'std','min','max')
        })\
        .reset_index()
    df_dowjones_nasdaq.columns = ['_'.join(col) for col in df_dowjones_nasdaq.columns.values]
    df_dowjones_nasdaq = df_dowjones_nasdaq.rename(columns={'Year_':'Year'})

    # base para modelo de arvore
    df_final = df_fuel_cons\
        .merge(df_uso_energia_or, on='Year', how='inner')\
        .merge(df_petroleo_year[['Year','Preco']], on='Year', how='inner')\
        .merge(df_fuel_exp, on='Year', how='inner')\
        .merge(df_dowjones_nasdaq, on='Year', how='inner')\
        .replace(0.0, np.nan)\
        .dropna(axis=0, thresh=0.5)\
        .dropna(axis=1, thresh=0.5)
    return df_final

@st.cache_data
def _df_brent_dowjones_nasdaq():
    _df = pd.read_csv('data/df_brent_dowjones_nasdaq_norm.csv', index_col='DATE')
    _df_dowjones = _df[['value_brent', 'value_dow_jones']]
    _df_dowjones.columns = ['Brent', 'Dow Jones']
    _df_nasdaq = _df[['value_brent', 'value_nasdaq']]
    _df_nasdaq.columns = ['Brent', 'Nasdaq']

    return _df_dowjones, _df_nasdaq

