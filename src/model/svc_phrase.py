# Tratamiento de datos
# ==============================================================================
from sklearn.preprocessing import StandardScaler
import warnings
import json
import opensmile
import pandas as pd
import numpy as np


# Preprocesado y modelado
# ==============================================================================
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate,cross_val_score, KFold
from sklearn.metrics import balanced_accuracy_score 
# ==============================================================================
warnings.filterwarnings('ignore')

        
# ========== "Load the data model" ==================================
# load data using Python JSON module


# apply the z-score method in Pandas using the .mean() and .std() methods
def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std
def z_standard(df):
    # create a scaler object
    std_scaler = StandardScaler()
    std_scaler
    # fit and transform the data
    df_std = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)
    return df_std

def split_data(path):
    path_csv = path
    df = pd.read_csv(path_csv)
    split = df["file"].str.split(r"\/|-", expand=True)                      
    df.loc[split[3] == "PATH", 'label'] = 1
    df.loc[split[3] == "NORM", 'label'] = 0   
    
    data = df.drop(columns=['file', 'start', 'end', 'type'])      
    #"all PATH(1) or NORM(0)")
    X = data.drop(columns=['label'], axis=1)
    y = data['label']
    X = X.astype(float).fillna(0.0)
    # print(data['label'].value_counts())

    #"========== normalization z_score =============")      
    X = z_standard(X)
    return X, y

nfold = 5
while nfold > 1:      
      X_train, y_train = split_data("data/csv/train_phrase_both_meta_data_fold"+str(nfold)+".csv")
      X_test, y_test = split_data("data/csv/test_phrase_both_meta_data_fold"+str(nfold)+".csv")
      
      # Creación del modelo SVM
      # ==============================================================================      
      model=[]
      data_processing = {'list':[],'model':[],'type_list':[], 'Kerner':[] , 'value_C':[], 'value_degree':[], 'Score':[], 'UAR':[]}      
      modelo = SVC(kernel='poly')
      
      scores_std = list()
      length=range(1,64, 4) 
      for c in length:            
            for deg in range(1,4):                  
                modelo.C = c
                modelo.degree = deg                
                modelo.fit(X_train, y_train)
                score= modelo.score(X_test, y_test) 
                y_pred = modelo.predict(X_test)
                uar = balanced_accuracy_score(y_test, y_pred)             
                
                data_processing['list'].append("phrase_fold"+str(nfold)+"")
                data_processing['model'].append("SVC")
                data_processing['type_list'].append("Phrase")
                data_processing['Kerner'].append("poly")
                data_processing['value_C'].append(c)
                data_processing['value_degree'].append(deg)
                data_processing['Score'].append(score*100)
                data_processing['UAR'].append(uar*100)                
      
      
      df = pd.DataFrame(data_processing, columns = ['list','model','type_list', 'Kerner' , 'value_C', 'value_degree', 'Score','UAR'])
      df.to_excel('svm_phrase.xlsx', sheet_name='svc_phrase', index=False)
      nfold -= 1 

# print("Load the data model")
