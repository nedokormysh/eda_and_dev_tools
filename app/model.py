import pandas as pd
import numpy as np
from typing import Type

# Метрика качества
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import make_scorer

# Разделение датасета
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import cross_validate

# Модели
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline, make_pipeline

import pickle

# from explainerdashboard import *

if __name__ == '__main__':
    train_prep = pd.read_csv('train_prep.csv')
    test_prep = pd.read_csv('test_prep.csv')

    y_train = train_prep['Rings']
    X_train = train_prep.drop(['Rings'], axis=1)

    y_test = test_prep['Rings']
    X_test = test_prep.drop(['Rings'], axis=1)

    cat_feat = ['Sex']
    num_feats = X_train.columns.drop('Sex').values

    num_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    cat_transformer = Pipeline(steps=[("onehot", OneHotEncoder())])

    data_transformer = ColumnTransformer(transformers=
                                         [('numerical', num_transformer, num_feats),
                                          ('categorical', cat_transformer, cat_feat)])

    model = KNeighborsRegressor()

    pipe_model = make_pipeline(data_transformer, model)

    grid = {
        'kneighborsregressor__n_neighbors': np.arange(2, 20, 2),
        'kneighborsregressor__weights': ['uniform', 'distance'],
        'kneighborsregressor__p': [1, 2]
    }

    gs = GridSearchCV(pipe_model,
                      grid,
                      scoring='neg_mean_squared_error',
                      cv=3,
                      n_jobs=-1,
                      verbose=2)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_


    # Save to file in the current working directory
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(best_model, file)


    # explainer = RegressionExplainer(best_model, X_test, y_test, permutation_metric='rmse')
    #
    # db = ExplainerDashboard(explainer)
    # db.to_yaml("dashboard.yaml", explainerfile="explainer.dill", dump_explainer=True)
