from explainerdashboard import *

import pickle
import pandas as pd

if __name__ == '__main__':

    pkl_filename = "pickle_model.pkl"

        # Load from file
    with open(pkl_filename, 'rb') as file:
        best_model = pickle.load(file)

    test_prep = pd.read_csv('test_prep.csv')

    y_test = test_prep['Rings']
    X_test = test_prep.drop(['Rings'], axis=1)

    explainer = RegressionExplainer(best_model, X_test, y_test, permutation_metric='rmse')

    db = ExplainerDashboard(explainer)
    db.to_yaml("dashboard.yaml", explainerfile="explainer.dill", dump_explainer=True)