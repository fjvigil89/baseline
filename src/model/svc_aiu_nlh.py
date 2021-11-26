# Tratamiento de datos
# ==============================================================================

from sklearn.preprocessing import StandardScaler
import warnings
import json
import opensmile
import pandas as pd
import numpy as np
from time import time

# Preprocesado y modelado
# ==============================================================================
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, precision_score, f1_score, recall_score, average_precision_score
# ==============================================================================
warnings.filterwarnings('ignore')

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

# Creación del modelo SVM
# # ==============================================================================            
data_processing = {
    'list':[],
    'model':[],
    'type_list':[],
     'Kerner':[] ,
     'value_C':[],
     'value_degree':[],
     'Score':[],
     'UAR':[],
     'precision': [],
     'f1Score': [],
     'recall': [],
     'average_precision': [],
     }      
modelo = SVC(kernel='poly')
matriz_uar= list()
matriz_precision = list()
matriz_f1Score=list()
matriz_recall=list()
matriz_average_precision=list()
nfold = 1
while nfold <= 1:   
    start_fold = time() #Aqui empieza el tiempo a contar
    X_train, y_train = split_data("data/csv/train_aiu_nlh_both_meta_data_fold"+str(nfold)+".csv")
    X_test, y_test = split_data("data/csv/test_aiu_nlh_both_meta_data_fold"+str(nfold)+".csv")  
    c_uar = list()
    c_precision = list()
    c_f1Score=list()
    c_recall=list()
    c_average_precision=list()
    # length=range(1,64, 4)       
    length=range(1,3)       
    for c in length:
        _uar=list()
        _precision = list()
        _f1Score=list()
        _recall=list()
        _average_precision=list()
        for deg in range(1,5):                  
            modelo.C = c
            modelo.degree = deg                
            modelo.fit(X_train, y_train)
            score= modelo.score(X_test, y_test) 
            y_pred = modelo.predict(X_test)
            y_score = modelo.decision_function(X_test)
                        
            uar = balanced_accuracy_score(y_test, y_pred)                             
            precision= precision_score(y_test, y_pred, average='macro')
            f1Score = f1_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')                
            average_precision = average_precision_score(y_test, y_score)
            
            
            _uar.append(uar)
            _precision.append(precision)
            _f1Score.append(f1Score)
            _recall.append(recall)
            _average_precision.append(average_precision)            
            
            
            data_processing['list'].append("phrase_fold"+str(nfold)+".csv")
            data_processing['model'].append("SVC")
            data_processing['type_list'].append("Phrase")
            data_processing['Kerner'].append("poly")
            data_processing['value_C'].append(c)
            data_processing['value_degree'].append(deg)
            data_processing['Score'].append(score*100)
            data_processing['UAR'].append(uar*100)                                
            data_processing['precision'].append(precision)
            data_processing['f1Score'].append(f1Score)
            data_processing['recall'].append(recall)
            data_processing['average_precision'].append(average_precision)

        c_uar.append(_uar)
        c_precision.append(_precision)
        c_f1Score.append(_f1Score)
        c_recall.append(_recall)
        c_average_precision.append(_average_precision)        
        
    
    matriz_uar.append(c_uar)
    matriz_precision.append(c_precision)
    matriz_f1Score.append(c_f1Score)
    matriz_recall.append(c_recall)
    matriz_average_precision.append(c_average_precision)       
    nfold += 1
    end_fold= time()-start_fold
    print("time: %0.10f." % end_fold)
    

matriz_uar = np.array(matriz_uar)
matriz_precision = np.array(matriz_precision)
matriz_f1Score = np.array(matriz_f1Score)
matriz_recall = np.array(matriz_recall)
matriz_average_precision = np.array(matriz_average_precision)  

media_uar = []
media_precision = []
media_f1Score = []
media_recall = []
media_average_precision =[]
std_uar = []
std_precision = []
std_f1Score = []
std_recall = []
std_average_precision=[]


for i in range(len(matriz_uar[0])):    
    for j in range(len(matriz_uar[0][0])):                                
        media_uar.append(np.mean([matriz_uar[0][i, j], matriz_uar[1][i, j] ]))
        std_uar.append(np.std([matriz_uar[0][i, j], matriz_uar[1][i, j] ]))
        # Precision
        media_precision.append(np.mean([matriz_precision[0][i, j], matriz_precision[1][i, j] ]))
        std_precision.append(np.std([matriz_precision[0][i, j], matriz_precision[1][i, j] ]))
        #f1_Score
        media_f1Score.append(np.mean([matriz_f1Score[0][i, j], matriz_f1Score[1][i, j] ]))
        std_f1Score.append(np.std([matriz_f1Score[0][i, j], matriz_f1Score[1][i, j] ]))
        #recall
        media_recall.append(np.mean([matriz_recall[0][i, j], matriz_recall[1][i, j] ]))
        std_recall.append(np.std([matriz_recall[0][i, j], matriz_recall[1][i, j] ]))
        #average
        media_average_precision.append(np.mean([matriz_average_precision[0][i, j], matriz_average_precision[1][i, j] ]))
        std_average_precision.append(np.std([matriz_average_precision[0][i, j], matriz_average_precision[1][i, j] ]))


df = pd.DataFrame(data_processing, columns = ['list',
    'model',
    'type_list',
    'Kerner' ,
    'value_C',
    'value_degree',
    'Score',
    'UAR',
    'precision',
    'f1Score',
    'recall',
    'average_precision'
    ])

df = df.assign(mean_uar="")
df = df.assign(std_uar="")
df = df.assign(mean_precision="")
df = df.assign(std_precision="")
df = df.assign(mean_f1Score="")
df = df.assign(std_f1Score="")
df = df.assign(mean_recall="")
df = df.assign(std_recall="")
df = df.assign(mean_average_precision="")
df = df.assign(std_average_precision="")

for i in range(len(media_uar)):    
    df['mean_uar'][i]= media_uar[i]
    df['std_uar'][i]=std_uar[i]
    #precision
    df['mean_precision'][i]= media_precision[i]
    df['std_precision'][i]=std_precision[i]
    #f1_Score
    df['mean_f1Score'][i]= media_f1Score[i]
    df['std_f1Score'][i]=std_f1Score[i]
    #recall
    df['mean_recall'][i]= media_recall[i]
    df['std_recall'][i]=std_recall[i]
    #average
    df['mean_average_precision'][i]= media_average_precision[i]
    df['std_average_precision'][i]=std_average_precision[i]



df.to_excel('data/xls/svc_aiu_nlh.xlsx', sheet_name='svc_aiu_nlh', index=False)
    