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
def _run_xgboost(df_final, path='models/xgb_model.pkl'):
    X, y = df_final.drop(columns=['Preco']), df_final['Preco']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if (os.path.isfile(path)):
        predict_pipeline = _get_xgb_model(path)
        return {
          'pipeline':predict_pipeline
        , 'mape':str(round(mean_absolute_percentage_error(y_test, predict_pipeline.predict(X_test))*100,2))+"%"
        , 'r2':round(r2_score(y_train, predict_pipeline.predict(X_train)), 4)
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