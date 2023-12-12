import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error,
                             r2_score)
import config

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

# cross validation 
def _run_cv_prophet(df_model, params, n_splits = 5, test_size = 12,):
    # Prevenindo logs do cmdstanpy
    import warnings
    import logging
    warnings.filterwarnings('ignore')
    logging.getLogger('fbprophet').setLevel(logging.ERROR) 
    logging.getLogger('prophet').setLevel(logging.ERROR) 
    logging.getLogger('hyperopt.tpe').setLevel(logging.ERROR) 
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    _res = []
    tscv =  TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    for train_index, test_index in tscv.split(df_model):
        X_train, X_test = df_model.iloc[train_index], df_model.iloc[test_index]
        y_train, y_test = df_model.y[train_index], df_model.y[test_index]
        hyper_model = Prophet(
                yearly_seasonality=params['yearly_seasonality'],
                daily_seasonality=params['daily_seasonality'],
                weekly_seasonality=params['weekly_seasonality'],
                seasonality_mode=params['seasonality_mode'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                changepoint_prior_scale=params["changepoint_prior_scale"],
                changepoint_range=params["changepoint_range"],
                holidays_prior_scale=params['holidays_prior_scale']
                )
        # adicionando feriados na execução do modelo
        hyper_model.add_country_holidays(country_name='BR')
        str_regressors = ''
        for reg in params['regressors']:
            str_regressors += f'{reg["regressor_name"]}_'
            hyper_model.add_regressor(reg["regressor_name"], prior_scale=reg["prior_scale"], mode=reg["mode"])
        hyper_model.fit(X_train)
        forecast_ = hyper_model.predict(X_train)
        pred = hyper_model.predict(X_test)
        
        _res.append({
            'n_splits':n_splits,
            'test_size':test_size,
            'regressors':str_regressors,
            'train_index':train_index,
            'test_index':test_index,
            'max_date_train':max(X_train.ds),
            'max_date_test':max(X_test.ds),
            'test_mape':mean_absolute_percentage_error(y_test, pred['yhat'].values),
            'all_series_mape':mean_absolute_percentage_error(y_train, forecast_["yhat"].values)
        })
    return _res, pd.DataFrame(_res).test_mape.mean()

def _get_best_params():
    import json
    with open(f'{config.BASE_PATH}/raw/best_params.json', 'r') as file:
        return json.load(file)

def _train_cv_prophet(_df):
    best_params = _get_best_params()
    train_end = pd.to_datetime('2023-01-01')
    X_train = _df.loc[_df.ds < train_end]
    X_test = _df.loc[_df.ds >= train_end]

    hyper_model = Prophet(
        yearly_seasonality=best_params["yearly_seasonality"],
        daily_seasonality=best_params["daily_seasonality"],
        weekly_seasonality=best_params["weekly_seasonality"],
        seasonality_mode=best_params['seasonality_mode'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        changepoint_prior_scale=best_params["changepoint_prior_scale"],
        changepoint_range=best_params["changepoint_range"],
        holidays_prior_scale=best_params['holidays_prior_scale'])

    for reg in best_params["regressors"]:
        hyper_model.add_regressor(reg["regressor_name"], prior_scale=reg["prior_scale"], mode=reg["mode"])
        
    hyper_model.add_country_holidays(country_name='BR')
            

    hyper_model.fit(X_train)
    pred = hyper_model.predict(X_test)
    forecast_ = hyper_model.predict(_df)
    return hyper_model, X_test, pred, X_train, forecast_
