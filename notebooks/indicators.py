from pandas.core.frame import DataFrame
import pydash as py_
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from ta import add_all_ta_features
from ta.trend import MACD
from ta.momentum import rsi
from ta.volume import OnBalanceVolumeIndicator as OBV
from scipy.stats import linregress

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.io as pio

class Indicators(object):
    """
    Implementation of all indicators to be used
    """
    def __init__(self, settings):
        self.settings = settings

    def gen_all(self, df, period = 31, macd_rsi_BB = False, rsi_window=14):
        df['RSI'] = self.get_rsi(df, window=rsi_window)
        df = self.get_MACD(df, period_short = 9, period_long = 26)
        df = self.get_bollinger_bands(df)
        crossovers = self.get_MACD_signals(df)
        bb_crossovers = self.get_BB_signals(df)
        suportes, df = self.get_supports(df)
        resistencias, df = self.get_resistences(df)
        if (macd_rsi_BB):
            return df, crossovers, bb_crossovers
        df['OBV'] = self.get_OBV(df)
        df = self.get_SMA(df)
        df = self.get_SMA(df, period_short = 9, period_long = 21)
        df = self.get_EMA(df)
        if (period != None):
            df = df[-int(period):]
        df = df.reset_index(drop=True)
        df = df.sort_values(['Datetime'])
        
        df = self.get_willians_percent(self.max_14(self.min_14(df)))
        
        hammers, df = self.get_hammers(df)
        df = df.sort_values(['Datetime'], ascending=True)
        df = self.get_tendencia_alta_baixa_divergencia(df)
        df = self.get_tendencia_alta(df)
        df = self.get_tendencia_baixa(df)
        #df = add_all_ta_features(df, open='Open', high="High", low="Low", close="Close", volume="Volume")
        ma_crossovers = self.get_EMA_SMA_signals(df)
        high_trend, low_trend, close_trend = self.get_tendencia_baixa_alta_lingress(df)
        return df, crossovers, ma_crossovers, bb_crossovers, hammers, suportes, resistencias, high_trend, low_trend, close_trend

    def get_OBV(self, df):
        return OBV(df.Close, df['Volume']).on_balance_volume()

    def get_rsi(self, df, window=14):
        """
        Calculates the Relative Strength Index for a coin_pair
        If the returned value is above 75, it's overbought (SELL IT!)
        If the returned value is below 25, it's oversold (BUY IT!)

        :param coin_pair: String literal for the market (ex: BTC-LTC)
        :type coin_pair: str
        :param period: Number of periods to query
        :type period: int
        :param unit: Ticker interval (one of: 'oneMin', 'fiveMin', 'thirtyMin', 'hour', 'week', 'day', and 'month')
        :type unit: str

        :return: RSI
        :rtype: float
        """
        return rsi(df.Close, window)

    def get_bollinger_bands(self, df, N = 20, k = 2.0):
        """
        Calculating bollinger bands where:
        
        :param df: dataframe with the prices
        :type: pd.DataFrame

        :param N: Moving Average periods to be used in calculations
        :type N: integer

        :param k: number of standard deviations to use in calcs
        :type k: float
        """

        df['Standard_Deviation'] = df['Close'].rolling(N).std()
        df['Middle_Band'] = df['Close'].rolling(N).mean()
        df['Upper_Band'] = df['Middle_Band'] + df['Standard_Deviation'] * k
        df['Lower_Band'] = df['Middle_Band'] - df['Standard_Deviation'] * k

        return df
    
    def get_BB_signals(self, df):
        """
        "Quando os preços das ações continuamente tocam a parte superior das Bandas Bollinger, os preços são ditos sobre-compra;
        inversamente, quando continuamente tocam na parte mais baixa, os preços são ditos sobre-venda, 
        o primeiro sinalizando momento de vender e o segundo de comprar."
        """
        bb_crossovers = df[["Datetime","Upper_Band", "Lower_Band", "Close"]]
        bb_crossovers.loc[:,['Crossover']] = False
        bb_crossovers.loc[:,['Signal']] = np.nan
        bb_crossovers.loc[:,['Binary_Signal']] = 0.0
        for i in range(len(bb_crossovers["Datetime"])):
            if bb_crossovers["Upper_Band"][i] <= bb_crossovers["Close"][i]:
                bb_crossovers.loc[i,['Binary_Signal']] = -1.0
                bb_crossovers.loc[i,['Signal']] = 'Sell'
                bb_crossovers.loc[i,['Crossover']] = True
            elif (bb_crossovers["Lower_Band"][i] >= bb_crossovers["Close"][i]):
                bb_crossovers.loc[i,['Binary_Signal']] = 1.0
                bb_crossovers.loc[i,['Signal']] = 'Buy'
                bb_crossovers.loc[i,['Crossover']] = True
        bb_crossovers = bb_crossovers.loc[bb_crossovers.Crossover==True]
        bb_crossovers
        return bb_crossovers

    def _get_text_tendencia_reg(self, value):
        if value < 0:
            return "Tendência de baixa - beta = " + str(round(value,2))
        if value > 0:
            return "Tendência de alta - beta = " + str(round(value,2))
        return "Tendência indefinida - beta = " + str(round(value,2))

    def get_tendencia_baixa_alta_lingress(self, df, n_days=21):
        """
            aqui, analisamos a tendencia de alta/baixa utilizando o beta da regressão linear em cima dos preços mais altos/baixos:
        """
        from scipy.stats import linregress

        df.loc[:,['date_id']] = df.index + 1
        data0 = df.copy()
        # high trend line

        data0 = df.tail(n_days)

        reg = linregress(
                            x=data0['date_id'],
                            y=data0['High'],
                            )

        data0.loc[:,['high_trend']] = reg[0] * data0['date_id'] + reg[1]

        high_trend = (f"Os Preços Máximos apontam {self._get_text_tendencia_reg(reg[0])}")
        # low trend line

        reg = linregress(
                            x=data0['date_id'],
                            y=data0['Low'],
                            )

        data0.loc[:,['low_trend']] = reg[0] * data0['date_id'] + reg[1]

        low_trend = (f"Os Preços Mínimos apontam {self._get_text_tendencia_reg(reg[0])}")

        #Close

        reg = linregress(
                            x=data0['date_id'],
                            y=data0['Close'],
                            )

        data0.loc[:,['close_trend']] = reg[0] * data0['date_id'] + reg[1]

        close_trend = (f"Os Preços de Fechamento apontam {self._get_text_tendencia_reg(reg[0])}")
        return high_trend, low_trend, close_trend

    def get_tendencia_alta(self, df):
        if ("divergencia_alta" in df.columns):
            tend_alta = ['Tend.Alta (Min d-2 > min(d-1))2', 'Tend.Alta (Min d-1 > min(d))']
            df[tend_alta] = df[tend_alta].fillna(False, axis=1)
            df['Tendencia_alta_3candle'] = df[tend_alta].all(True)
        return df

    def get_tendencia_baixa(self, df):
        if ("divergencia_baixa" in df.columns):
            tend_baixa = ['Tend.Baixa(Máx d-1 & d-2)', 'Tend.Baixa(Máx d & d-1)']
            df[tend_baixa] = df[tend_baixa].fillna(False, axis=1)
            df['Tendencia_baixa_3candle'] = df[tend_baixa].all(True)
        return df

    def get_tendencia_alta_baixa_divergencia(self, df):
        df = df.sort_values(['Datetime'])
        df[['divergencia_alta']] = 0
        df[['divergencia_baixa']] = 0
        for i, row in df.iterrows():
            if (i == 0):
                row_m2 = row
            elif (i == 1):
                row_m1 = row
            else:
                df.loc[i,['Tend.Alta (Min d-2 > min(d-1))2']] = (row_m2['Low'] > row_m1['Low'])
                df.loc[i,['Tend.Alta (Min d-1 > min(d))']] = (row_m1['Low'] > row['Low'])
                df.loc[i,['Tend.Alta (Hist d-2 < histd-1)']] = (row_m2['MACD_Histogram'] < row_m1['MACD_Histogram'])
                df.loc[i,['Tend.Alta (Hist d-1 < histd)2']] = (row_m1['MACD_Histogram'] < row['MACD_Histogram'])
                df.loc[i,['MACD < 0']] = (row['MACD'] < 0)
                df.loc[i,['Tend.Baixa(Máx d-1 & d-2)']] = (row_m2['High'] < row_m1['High'])
                df.loc[i,['Tend.Baixa(Máx d & d-1)']] = (row_m1['High'] < row['High'])
                df.loc[i,['Tend.Baixa (Hist d & d-1)']] = (row_m1['MACD_Histogram'] > row['MACD_Histogram'])
                df.loc[i,['Tend.Baixa (Hist d-2 & d-1)']] = (row_m2['MACD_Histogram'] > row_m1['MACD_Histogram'])
                row_m1 = row
                row_m2 = row_m1
                if ((df.loc[i-2, ['Low']][0]) > ((df.loc[i-1, ['Low']][0])) > (df.loc[i, ['Low']][0])):
                    if ((df.loc[i-2, ['MACD_Histogram']][0]) < ((df.loc[i-1, ['MACD_Histogram']][0])) < (df.loc[i, ['MACD_Histogram']][0])):
                        if (df.loc[i, ['Support']][0] == 1 and df.loc[i, ['MACD']][0]<0):
                            df.loc[i,['divergencia_alta']] = 1
                if ((df.loc[i-2, ['High']][0]) < ((df.loc[i-1, ['High']][0])) < (df.loc[i, ['High']][0])):
                    if ((df.loc[i-2, ['MACD_Histogram']][0]) > ((df.loc[i-1, ['MACD_Histogram']][0])) > (df.loc[i, ['MACD_Histogram']][0])):
                        df.loc[i,['divergencia_baixa']] = 1
        return df.sort_values(['Datetime'], ascending=True)
        
    def max_14(self, df):
        df['MAX_14'] = df['High'].rolling(14).max()
        return df

    def min_14(self, df):
        df['MIN_14'] = df['Low'].rolling(14).min()
        return df
    
    def get_willians_percent(self, df):
        df['Willians_percent'] = ((df['MAX_14'] - df['Close'])/(df['MAX_14']-df['MIN_14']))*-100
        df['Willians Buy'] = [1 if x<-80 else 0 for x in df.Willians_percent]
        df['Willians Sell'] = [1 if x>-10 else 0 for x in df.Willians_percent]
        return df
    
    def get_hammers(self, df):
        hammers = []
        df[['Hammer']] = 0
        for i in range(2, len(df)):
            if ((df.iloc[i - 2].Close - df.iloc[i - 2].Open) < 0):
                if ((df.iloc[i - 1].Close - df.iloc[i - 1].Open) < 0):
                    if (((df.iloc[i].Open - df.iloc[i].Low) / (df.iloc[i].High - df.iloc[i].Low)) > 0.666):
                        if ((df.iloc[i].High - df.iloc[i].Low) > (2 * (df.iloc[i].Open - df.iloc[i].Close)) ):
                            df.loc[i,['Hammer']] = 1
                            hammers.append(df.iloc[i])
        return pd.DataFrame(hammers), df
    
    def get_supports(self, df):
        suportes = []
        df['Smin_21'] = df.Low.rolling(21).min()
        df['Support'] = [self._test_tuple(x, 'Low', 'Smin_21') for x in df.itertuples()]
        return df.query('Support == 1'), df
    
    def get_resistences(self, df):
        df['Rmax_21'] = df.High.rolling(21).max()
        df['Resistance'] = [self._test_tuple(x, 'High', 'Rmax_21') for x in df.itertuples()]
        return df.query('Resistance == 1'), df
    
    def get_SMA(self, df, period_short = 5, period_long = 20, period_longest = 200):
        df["SMA_"+str(period_short)] = df.Close.rolling(period_short).mean()
        df["SMA_"+str(period_long)] = df.Close.rolling(period_long).mean()
        df["SMA_"+str(period_longest)] = df.Close.rolling(period_longest).mean()
        return df

    def get_EMA(self, df, period_short = 9, period_long = 21, period_longest = 200):
        df["EMA_"+str(period_short)] = df.Close.ewm(span=int(period_short), adjust=False).mean()
        df["EMA_"+str(period_long)] = df.Close.ewm(span=int(period_long), adjust=False).mean()
        df["EMA_"+str(period_longest)] = df.Close.ewm(span=int(period_longest), adjust=False).mean()
        return df

    def get_MACD(self, df, period_short, period_long):
        df["EMA_Short"] = df.Close.ewm(span=int(period_short), adjust=False).mean()
        df["EMA_Long"] = df.Close.ewm(span=int(period_long), adjust=False).mean()
        df['12MME'] = df.Close.ewm(span=12, adjust=False).mean()
        df['9MME'] = df.Close.ewm(span=9, adjust=False).mean()
        df['26MME'] = df.Close.ewm(span=26, adjust=False).mean()
        df['MACD'] = df.Close.ewm(span=12, adjust=False).mean() - df.Close.ewm(span=26, adjust=False).mean()
        df['Posição Alta MACD'] = [1 if x>0 else 0 for x in df.MACD]
        df['Posição Baixa MACD'] = [1 if x<0 else 0 for x in df.MACD]
        df['MACD_Signal_line'] = df.MACD.ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal_line']
        df['MACD_Signal'] = ""
        for i in range(1,len(df["MACD_Signal"])):
            if (df.MACD[i] > df.MACD_Signal_line[i]):
                df.loc[i, ['MACD_Signal']] = "Buy"
            elif (df.MACD[i] < df.MACD_Signal_line[i]):
                df.loc[i, ['MACD_Signal']] = "Sell"
        df["Prev_MACD_Signal"] = df['MACD_Signal'].shift(1)
        for i in range(1,len(df["MACD_Signal"])):
            if (df.MACD_Signal[i] == df.Prev_MACD_Signal[i]):
                df.loc[i, ['MACD_Signal']] = ""
        df.loc[0, ['MACD_Signal']] = ""
        df.drop(columns="Prev_MACD_Signal", inplace=True)
        return df
    
    def get_MACD_signals(self, df):
        crossovers = pd.DataFrame()
        idx = df['Datetime']
        crossovers['Price'] = [i for i in df['Close']]
        crossovers["MACD"] = df["MACD"]
        crossovers['MACD_Signal_line'] = df['MACD_Signal_line']
        crossovers['position'] = crossovers["MACD"] >= crossovers["MACD_Signal_line"]
        crossovers.index = idx
        crossovers['pre-position'] = crossovers['position'].shift(1)
        crossovers['Crossover'] = np.where(crossovers['position'] == crossovers['pre-position'], False, True)
        crossovers = crossovers.reset_index()
        crossovers.loc[0,['Crossover']] = False
        #print(crossovers)

        crossovers = crossovers.loc[crossovers['Crossover'] == True]
        crossovers = crossovers.reset_index()
        crossovers = crossovers.drop(['position', 'pre-position', 'Crossover'], axis=1)
        crossovers['Signal'] = np.nan
        crossovers['Binary_Signal'] = 0.0
        for i in range(len(crossovers["MACD"])):
            if crossovers["MACD"][i] > crossovers["MACD_Signal_line"][i]:
                crossovers.loc[i,['Binary_Signal']] = 1.0
                crossovers.loc[i,['Signal']] = 'Buy'
            else:
                crossovers.loc[i,['Binary_Signal']] = -1.0
                crossovers.loc[i,['Signal']] = 'Sell'
        
        crossovers = (crossovers[["Datetime","Price", "MACD","MACD_Signal_line","Signal","Binary_Signal"]])
        crossovers['Sinal de Compra (MACD)'] = [1 if x=='Buy' else 0 for x in crossovers.Signal]
        crossovers['Sinal de Venda (MACD)'] = [1 if x=='Sell' else 0 for x in crossovers.Signal]
        return crossovers
 
    def get_EMA_SMA_signals(self, df):
        ma_crossovers = df[["Datetime","EMA_9", "SMA_21"]]
        ma_crossovers.loc[:,['position']] = ma_crossovers["EMA_9"] >= ma_crossovers["SMA_21"]
        ma_crossovers.loc[:,['pre-position']] = ma_crossovers['position'].shift(1)
        ma_crossovers.loc[:,['Crossover']] = np.where(ma_crossovers['position'] == ma_crossovers['pre-position'], False, True)
        ma_crossovers.loc[0,['Crossover']] = False
        ma_crossovers.loc[:,['Signal']] = np.nan
        ma_crossovers.loc[:,['Binary_Signal']] = 0.0
        for i in range(len(ma_crossovers["Datetime"])):
            if ma_crossovers["EMA_9"][i] > ma_crossovers["SMA_21"][i]:
                ma_crossovers.loc[i,['Binary_Signal']] = 1.0
                ma_crossovers.loc[i,['Signal']] = 'Buy'
            else:
                ma_crossovers.loc[i,['Binary_Signal']] = -1.0
                ma_crossovers.loc[i,['Signal']] = 'Sell'
        ma_crossovers = ma_crossovers.drop(['position','pre-position'], axis=1).loc[ma_crossovers.Crossover==True]
        return ma_crossovers

    def _test_tuple(self, _tuple, col1, col2):
        if (getattr(_tuple, col1) == getattr(_tuple, col2)):
            return 1
        return 0
    
     
def generate_graph(df, crossovers, just_return = False, just_candles = True, height = 800, width=800):
    INCREASING_COLOR = '#3D9970'
    DECREASING_COLOR = '#FF4136'

    df.index = df.Datetime

    fig = make_subplots(rows=5, cols=1)
    fig = fig.update_layout(title='Candle Plot w/ Plotly', title_yanchor='middle')
    fig = fig.add_trace(go.Candlestick(x=df.index,
                                       open=df['Open'],
                                       high=df['High'],
                                       low=df['Low'],
                                       close=df['Close']),
                        row=1, col=1
                       )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['Upper_Band'],#
               name='Upper BB',  
               marker=dict(color=['black'])
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Lower_Band'],#
               name='Lower BB', 
               marker=dict(color=['black'])
        ),
        row=1, col=1
    )
    
    fig = fig.update_layout(xaxis_rangeslider_visible=False, xaxis={'visible':False, 'showticklabels':False})
    # volume bar chart

    colors = []

    for i in range(len(df.Close)):
        if i != 0:
            if df.Close[i] > df.Close[i-1]:
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)


    # add vol to graph
    fig.add_trace(
        go.Bar(x=df.index, y=df.Volume,
               marker=dict(color=colors),#
               name='Volume'),
        row=2, col=1

    )
    
    if (just_candles and just_return):
        return fig
    
    # OBV
    fig.add_trace(
        go.Scatter(x=df.index, y=df.OBV,
               marker=dict(color=colors),#
               name='OBV'),
        row=2, col=1

    )
    
    fig = fig.update_layout(xaxis={'visible':False, 'showticklabels':False}, yaxis={'visible':False, 'showticklabels':False})
    
    colors_macd = []

    for i in range(len(df.MACD_Histogram)):
        if i != 0:
            if df.MACD_Histogram[i] >= 0:
                colors_macd.append(INCREASING_COLOR)
            else:
                colors_macd.append(DECREASING_COLOR)
        else:
            colors_macd.append(DECREASING_COLOR)
    
    # add MACD HIST to graph
    fig.add_trace(
        go.Bar(x=df.index, y=df.MACD_Histogram,
               marker=dict(color=colors_macd),#
               name='MACD Histogram'),
        row=3, col=1

    )
    
    # add 12-EMA
    fig.add_trace(
        go.Scatter(x=df.index, y=df.EMA_9,#
               name='EMA_9'),
        row=4, col=1

    )
    
    # add 26-EMA
    fig.add_trace(
        go.Scatter(x=df.index, y=df.SMA_21,#
               name='SMA_21'),
        row=4, col=1

    )
    
    fig = fig.update_layout(xaxis={'visible':False, 'showticklabels':False}, yaxis={'visible':False, 'showticklabels':False})
    
    # add MACD markers
    fig.add_trace(
        go.Scatter(x=df.Datetime, y=df.MACD, 
               marker=dict(color="black"), name='MACD'
              ),
        row=5, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.Datetime, y=df.MACD_Signal_line, 
               marker=dict(color="blue"), name='MACD Signal'
              ),
        row=5, col=1
    )
    
    fig = fig.update_layout(yaxis={'visible':False, 'showticklabels':False})
    
    fig = fig.update_layout(xaxis_rangeslider_visible=False)
    
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,)
    
    if (just_return): 
        return fig
    
    fig.show()