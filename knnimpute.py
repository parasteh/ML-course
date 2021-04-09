import pandas as pd

#loading the enriched data

data_df = pd.read_csv('train_df_befor_imputing.csv')
input_df =  data_df[['PROVINCE','VALUE_PER_UNIT','LAND_DETAIL_TYPE','SOIL_TEXTURE','TILE_DRAINAGE','IRRIGATION','LONGITUDE','LATITUDE','is_urban']]
onehot_df = pd.get_dummies(input_df, prefix_sep='_',columns=['PROVINCE','LAND_DETAIL_TYPE'], drop_first=True)
# Imputation #1:
# Using KNN approach to do the imputation
from sklearn.impute import KNNImputer
from missingpy import MissForest
import datawig

def knn_imputing(data):
  # code me !

  imputer = KNNImputer()
  data = pd.DataFrame(imputer.fit_transform(data),columns = data.columns)
  return data

def rf_imputing(data):
  #code me !
  # Make an instance and perform the imputation
  imputer = MissForest(verbose=True)
  X = data.drop('VALUE_PER_UNIT', axis=1)
  X_imputed = imputer.fit_transform(X)
  # X_imputed['VALUE_PER_UNIT'] =  data['VALUE_PER_UNIT']

  return X_imputed

def nn_imputing(data):
  #code me !
  df_with_missing_imputed = datawig.SimpleImputer.complete(data)

  return df_with_missing_imputed

train_knn_imputed =  knn_imputing(onehot_df)

train_knn_imputed.to_csv('final_knn_imputed.csv')