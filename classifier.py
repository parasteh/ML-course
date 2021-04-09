import xgboost
from sklearn.linear_model import SGDRegressor, BayesianRidge, ARDRegression, PassiveAggressiveRegressor, \
    TheilSenRegressor, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pandas as pd
import numpy as np



train_df = pd.read_csv('X.csv')

train_df = train_df.sample(5000)






X = train_df.drop('VALUE_PER_UNIT', axis=1)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, train_df.VALUE_PER_UNIT, test_size=0.2, random_state=42)
y = y_train


classifiers = [
    #SVR(),
    #     SGDRegressor(),
    #      ,
    #      BayesianRidge()
    #     linear_model.LassoLars(),
    #      ARDRegression()
    # PassiveAggressiveRegressor(),
    # TheilSenRegressor(),
    #     LinearRegression(),
    xgboost.XGBRegressor(colsample_bytree=0.6, gamma=0.1, min_child_weight=1.5, learning_rate=0.07, max_depth=5,
                         n_estimators=1000, reg_alpha=0.01, reg_lambda=1e-05, subsample=0.95)

]



def mpe(pred, real):
    y_test, y_pred = np.array(real), np.array(pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100


mpe_res = []
for item in classifiers:
    print(item)
    clf = item
    clf.fit(X_train, y)

    mpe_res.append(mpe(clf.predict(X_train), y))

print(mpe_res)