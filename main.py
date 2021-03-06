import pandas as pd
import numpy as np
from missingpy import MissForest
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import xgboost

from tune_sklearn import TuneGridSearchCV, TuneSearchCV

train_df = pd.read_csv('X.csv')



def rf_imputing(data):
  #code me !
  # Make an instance and perform the imputation
  imputer = MissForest(verbose=True)
  X = data.drop('VALUE_PER_UNIT', axis=1)
  X_imputed = imputer.fit_transform(X)
  # X_imputed['VALUE_PER_UNIT'] =  data['VALUE_PER_UNIT']
  return X_imputed

# train_rf_imputed = rf_imputing(train_df)

X = train_df.drop('VALUE_PER_UNIT', axis=1)


# rf_imputed = pd.DataFrame(train_rf_imputed, columns=[X.columns])
# X_imputed = rf_imputed.drop(columns=[ 'Unnamed: 0'], axis= 1)



scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, train_df.VALUE_PER_UNIT, test_size=0.2, random_state=42)


# # XGBoost
# xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
#     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)

xgb_model = xgboost.XGBRegressor()

#for tuning parameters
parameters_for_testing = {
   'colsample_bytree':[0.4,0.6,0.8],
   'gamma':[0,0.03,0.1,0.3],
   'min_child_weight':[1.5,6,10],
   'learning_rate':[0.1,0.07],
   'max_depth':[3,5],
   'n_estimators':[10,50,100, 500, 1000],
   'reg_alpha':[1e-5, 1e-2,  0.75],
   'reg_lambda':[1e-5, 1e-2, 0.45],
   'subsample':[0.6,0.95]
}

t_search = TuneSearchCV(
    xgb_model,
    param_distributions=parameters_for_testing,
    n_trials=3,
    early_stopping=True,
    use_gpu=True
    # Commented out for testing on github actions,
    # but this is how you would use gpu
)

# gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,iid=False, verbose=10,scoring='neg_mean_squared_error')
t_search.fit(X_train,y_train)
print (t_search.scorer_)
print('best params')
print (t_search.best_params_)
print('best score')
print (t_search.best_score_)

final_xgb = xgboost.XGBRegressor(colsample_bytree= 0.6, gamma= 0.1, min_child_weight= 1.5, learning_rate= 0.07, max_depth= 5, n_estimators= 1000, reg_alpha= 0.01, reg_lambda= 1e-05, subsample= 0.95)

trained = final_xgb.fit(X_train,y_train)


y_pred = trained.predict(X_test)
def mean_absolute_percentage_error(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return  np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(mean_absolute_percentage_error(y_test, y_pred))

