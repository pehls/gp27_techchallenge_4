import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error,
                             r2_score)


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