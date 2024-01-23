import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error,
                             r2_score)
import config
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os
import streamlit as st
import numpy as np

@st.cache_resource
def _train_simple_prophet(_df):
    _model = Prophet()

    train_end = pd.to_datetime('2023-01-01')
    X_train = _df.loc[_df.ds < train_end]
    X_test = _df.loc[_df.ds >= train_end]

    _model.fit(X_train)
    forecast_ = _model.predict(X_train)
    pred = _model.predict(X_test)
    _df = pd.concat([_df, _model.predict(_df)])
    return _model, X_test, pred, X_train, forecast_

@st.cache_resource
def _run_xgboost(df_final, path='models/xgb_model.pkl', predict=False):
    X, y = df_final.drop(columns=['Preco']), df_final['Preco']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if (os.path.isfile(path)) and not(predict):
        predict_pipeline = _get_xgb_model(path)
        return {
          'pipeline':predict_pipeline
        , 'mape':str(round(mean_absolute_percentage_error(y_test, predict_pipeline.predict(X_test))*100,2))+"%"
        , 'r2':round(r2_score(y_train, predict_pipeline.predict(X_train)), 4)
        , 'predictions':predict_pipeline.predict(df_final)
        }
    
    if (os.path.isfile(path)) and not(predict):
        predict_pipeline = _get_xgb_model(path)
        return {
          'pipeline':predict_pipeline
        , 'mape':str(round(mean_absolute_percentage_error(df_final['Preco'], predict_pipeline.predict(df_final.drop(columns=['Preco'])))*100,2))+"%"
        , 'r2':round(r2_score(df_final['Preco'], predict_pipeline.predict(df_final.drop(columns=['Preco']))), 4)
        , 'predictions':predict_pipeline.predict(df_final.drop(columns=['Preco']))
        }

    numeric_features = list(set(X.columns) - set(['Year']))
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )

    categorical_features = ["Year"]
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
            # , ("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    predict_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", XGBRegressor(seed=42))]
    )

    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

    predict_pipeline.fit(X_train, y_train)
    return {
          'pipeline':predict_pipeline
        , 'mape':str(round(mean_absolute_percentage_error(y_test, predict_pipeline.predict(X_test))*100,2))+"%"
        , 'r2':round(r2_score(y_train, predict_pipeline.predict(X_train)), 4)
        }
@st.cache_data
def _get_tree_importances(_predict_pipeline):
    model = _predict_pipeline['regressor']
    df_importances = pd.DataFrame([_predict_pipeline[:-1].get_feature_names_out(), model.feature_importances_], index=['Features','Importance']).T
    df_importances = df_importances.loc[df_importances.Importance > 0.0001].sort_values('Importance', ascending=False)
    return df_importances

@st.cache_resource
def _get_xgb_model(path='models/xgb_model.pkl'):
    return joblib.load(path)

@st.cache_data
def check_causality(data : pd.DataFrame, list_of_best_features : list, y_col : str, threshold : float = 0.05) -> pd.DataFrame:
    data = data.dropna(axis=0, thresh=0.7).dropna(axis=1)
    from statsmodels.tsa.stattools import grangercausalitytests

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = pd.DataFrame(imp_mean.fit_transform(data), columns=data.columns)

    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    maxlag=1
    test = 'ssr_chi2test'

    def grangers_causation_matrix(data, variables, focus = None, test=test, verbose=False):   
        variables = list(set(variables))
        if (type(focus) != type(None)): 
            df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
        else:
            df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=focus)
        for c in df.columns:
            for r in df.index:
                try:
                    test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
                    p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                    if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
                    min_p_value = np.min(p_values)
                except:
                    print(f'r = {r}, c = {c}')
                    min_p_value = 99
                df.loc[r, c] = min_p_value
        df.index = [var + '_y' for var in df.index]
        df = df.reset_index()
        df.columns = ['variable'] + [var + '_x' for var in variables]
        return df
    g_matrix = (grangers_causation_matrix(data.dropna(), variables = list_of_best_features + [y_col], focus=[y_col]))
    #display(g_matrix)
    #g_matrix.columns = ['variable'] + [x + '_x' for x in list_of_best_features + ['QT_VOLUME_SELL_THROUGH']]
    
    # g_matrix = g_matrix.loc[g_matrix['QT_VOLUME_SELL_THROUGH_x'] < threshold]
    # var_y = list(set([x.replace('_y','') for x in g_matrix.variable]))
    g_matrix = g_matrix.loc[g_matrix['variable']==f'{y_col}_y'].T.reset_index()
    g_matrix.columns = g_matrix.iloc[0]
    g_matrix = g_matrix[1:].reset_index(drop=True)
    g_matrix = g_matrix.loc[g_matrix[f'{y_col}_y'] < threshold]
    g_matrix.variable = [x.replace('_x','') for x in g_matrix.variable]
    g_matrix.columns = ['Variable','Sign.']
    g_matrix = g_matrix.sort_values(f'Sign.', ascending=True)
    # g_matrix.index = g_matrix['Variable']
    return g_matrix[['Variable','Sign.']]