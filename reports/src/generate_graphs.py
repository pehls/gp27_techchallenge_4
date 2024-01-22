import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive, SeasonalWindowAverage
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import src.get_data as get_data 


def _grafico_historico(df):
    fig = go.Figure()

    fig = px.line(
        df, 
        x='Date', y='Preco'
        #, custom_data=['year', config.DICT_Y[stat][0], 'country_code']
    )

    # hide axes
    fig.update_xaxes(visible=True, title='')
    fig.update_yaxes(visible=True,
                    gridcolor='black',zeroline=True,
                    showticklabels=True, title=''
                    )
    fig.update_layout(
        hovermode='x unified',
    )

    # fig.update_traces(
    #     hovertemplate="""
    #     <b>Metr.:</b> %{customdata[1]} 
    #     <b>Med.:</b> %{customdata[2]} 
    #     <b>Mediana:</b> %{customdata[3]} 
    #     <b>Max.:</b> %{customdata[4]} 
    #     """
    # )

    # strip down the rest of the plot
    fig.update_layout(
        showlegend=True,
        # plot_bgcolor="black",
        margin=dict(l=10,b=10,r=10)
    )
    return fig

def _seasonal_decompose(series, period=5):
    result = seasonal_decompose(series, model='additive', period=period)
    return result.plot()

def _adf(df):
    df.index = pd.to_datetime(df['Date'], dayfirst=True)
    ma = df.Preco.rolling(12).mean()

    fig, ax = plt.subplots()
    df.plot(ax=ax, legend=False)
    ma.plot(ax=ax, legend=False)

    return fig, df.Preco.values

def _adf_diff(df):
    df.index = pd.to_datetime(df['Date'], dayfirst=True)

    df['Preco_diff'] = df.Preco.astype(float).diff(1)

    ma = df.Preco.rolling(12).mean()
    std = df.Preco.rolling(12).std()

    fig, ax = plt.subplots()
    df.Preco_diff.plot(ax=ax, legend=False)
    ma.plot(ax=ax, legend=False)
    std.plot(ax=ax, legend=False)

    return fig, df.dropna()['Preco_diff'].values

def _models_ts():
    train, test, h = get_data._get_data_for_models_ts()
    
    model_all = StatsForecast(models=[
        SeasonalNaive(season_length=7),
        SeasonalWindowAverage(season_length=7, window_size=2),
        AutoARIMA(season_length=7)], freq='D', n_jobs=-1)
    
    model_all.fit(train)

    forecast_all = model_all.predict(h=h)
    forecast_all = forecast_all.reset_index().merge(test, on=['ds', 'unique_id'], how='left')
    forecast_all.dropna(inplace=True)

    mape_seas_naive = mean_absolute_percentage_error(forecast_all['y'].values, forecast_all['SeasonalNaive'].values)
    mape_seas_wa = mean_absolute_percentage_error(forecast_all['y'].values, forecast_all['SeasWA'].values)
    mape_arima = mean_absolute_percentage_error(forecast_all['y'].values, forecast_all['AutoARIMA'].values)
    graph = model_all.plot(train, forecast_all, unique_ids=['PETR4'], engine='plotly') 

    return graph, mape_seas_naive, mape_seas_wa, mape_arima

def _plot_trials(trials_df, hyperparam_1, hyperparam_2):
    from plotly import graph_objects as go
    fig = go.Figure(
        data=go.Contour(
            z=trials_df.loc[:, "loss"],
            x=trials_df.loc[:, hyperparam_1],
            y=trials_df.loc[:, hyperparam_2],
            contours=dict(
                showlabels=True,  # show labels on contours
                labelfont=dict(size=12, color="white",),  # label font properties
            ),
            colorbar=dict(title="loss", titleside="right",),
            colorscale='Hot',
            hovertemplate="loss: %{z}<br>"+hyperparam_1+": %{x}<br>"+hyperparam_2+": %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title=hyperparam_1,
        yaxis_title=hyperparam_2,
        title={
            "text": f"{hyperparam_1} vs. {hyperparam_2} ",
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )
    return fig

def _plot_conflitos_paises(df, order={}):
    df['Date'] = [x.replace(day=1) for x in df.Date]
    df['Year'] = [x.year for x in df.Date]
    df = df\
    .groupby(['country'])\
    .agg({
          'fatalities':'sum'
        , 'event_id_cnty':'nunique'
        })\
        .sort_values('fatalities', ascending=False)\
        .reset_index()\
        .head(10)
    fig = px.bar(
        df,
        x='country', y='fatalities',
        color='country',
        title="Fatalidades por País (2020-2023)",
        color_discrete_map={
              'Ignorado':'purple'
            , 'Não' : 'red'
            , "Não sabe" : 'goldenrod'
            , 'Sim':'blue'
            }, text_auto=True,
        category_orders=order,
        hover_data=['event_id_cnty']
    )
    fig.update_layout(
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False
        ),
        xaxis=dict(
            title='País',
            showgrid=False,
            showline=False,
            showticklabels=True
        )
    )
    fig.for_each_trace(lambda t: t.update(texttemplate = t.texttemplate + ''))

    return fig

def _plot_conflitos_tipo(df):
    df['Date'] = [x.replace(day=1) for x in df.Date]
    df = df\
    .groupby(['Date','event_type'])\
    .agg({
          'fatalities':'sum'
        , 'event_id_cnty':'sum'
    }).reset_index()
    
    fig = px.line(
        df, 
        x='Date', y='event_id_cnty', color='event_type'
        #, custom_data=['year', config.DICT_Y[stat][0], 'country_code']
    )

    # hide axes
    fig.update_xaxes(visible=True, title='')
    fig.update_yaxes(visible=True,zeroline=True,
                    showticklabels=True, title=''
                    )
    fig.update_layout(
        hovermode='x unified',
    )

    # strip down the rest of the plot
    fig.update_layout(
        showlegend=True,
        # plot_bgcolor="black",
        margin=dict(l=10,b=10,r=10)
    )

    return fig

def _plot_conflitos_tipo_e_petroleo(df_conflitos_mundiais):
    cols_to_plot = ['Fatalities in Battles','Fatalities in Explosions/Remote Violence','Fatalities in Violence against civillians', 'Qtt Battles','Qtt Explosions/Remote violence','Qtt Violence against civilians', 'Preco','Total Qtt','Total Fatalities']
    fig = go.Figure()

    for col in cols_to_plot:
        fig.add_trace(go.Scatter(
            x=df_conflitos_mundiais['Date'], y=df_conflitos_mundiais["minmax_"+col],
            mode='lines', yaxis='y', name=col.replace('minmax_','')
            #, custom_data=['year', config.DICT_Y[stat][0], 'country_code']
            )
        )

    # hide axes
    fig.update_layout(
        hovermode='x unified',
    )
    return fig

def _plot_correlation_matrix(df_correlacoes):
    fig = px.imshow(df_correlacoes, text_auto=True, aspect='auto',)
    fig.update_layout(
        minreducedheight=600
    )
    return fig

def _plot_energy_use(df_uso_energia):
    cols_to_plot = list(set(df_uso_energia.columns)-set(['Country Name']))
    fig = go.Figure()

    
    # fig.add_trace(go.Scatter(
    #     x=df_uso_energia['Year'], y=df_uso_energia['Country Name'],
    #     mode='lines', yaxis='y'#, name=col
    #     #, custom_data=['year', config.DICT_Y[stat][0], 'country_code']
    #     )
    # )
    fig = px.line(
        df_uso_energia, 
        x='Year', y='value', color='Country Name'
        #, custom_data=['year', config.DICT_Y[stat][0], 'country_code']
    )

    # hide axes
    fig.update_layout(
        hovermode='x unified',
    )
    return fig

def _plot_top10_recent_energy_use(df_uso_energia_top10):
    fig = px.bar(
        df_uso_energia_top10
        , x='Total En. Use 11-15', y='Country Name', orientation='h'
        , text='Total En. Use 11-15', title='Uso de Energia Primária (em kg de óleo equivalente per capita)'
    )
    fig.update_layout(
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True
        ),
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True
        )
    )
    return fig

def _plot_df_importances(df_importances):
    fig = px.bar(
        df_importances
        , x='Features', y='Importance'
        , title='Importância das Variáveis no Modelo'
    )
    fig.update_layout(
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True
        ),
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True
        )
    )
    return fig