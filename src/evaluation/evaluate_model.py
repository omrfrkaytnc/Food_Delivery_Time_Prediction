from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor




rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, 1.0],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

gb_params = {'n_estimators': [50, 100, 150],
             'learning_rate': [0.01, 0.1, 0.2],
             'max_depth': [3, 5, 10]}

xgb_params = {'n_estimators': [50, 100, 150],
              'learning_rate': [0.01, 0.1, 0.2],
              'max_depth': [3, 5, 10]}




classifiers = [('RF', RandomForestRegressor(), rf_params),
               ('XGBoost', XGBRegressor(use_label_encoder=False, eval_metric='logloss'), xgb_params),
               ('GB', GradientBoostingRegressor(), gb_params)]



def hyperparameter_optimization(X, y, cv=3, scoring="r2"):
    print("Hyperparameter Optimization....")
    best_models = {}
    regressors = [
        ("RF", RandomForestRegressor(), rf_params),
        ("GB", GradientBoostingRegressor(), gb_params),
        ("XGB", XGBRegressor(use_label_encoder=False, eval_metric='logloss'), xgb_params)
    ]
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        cv_results = cross_validate(regressor, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)



"""
Hyperparameter Optimization....
########## RF ##########
r2 (Before): 0.83
r2 (After): 0.8357
RF best params: {'max_depth': 15, 'max_features': 1.0, 'min_samples_split': 20, 'n_estimators': 300}

########## GB ##########
r2 (Before): 0.7747
r2 (After): 0.8337
GB best params: {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 50}
########## XGB ##########
r2 (Before): 0.818
r2 (After): 0.8351
XGB best params: {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 50}
"""

