# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
arreglo = [[[79.84862132085888, 65.26861725310167, 62.04491966179504, 52.386901821774124], [81.44812156782986, 70.43612168405149, 66.86811750007264, 54.56460470116512]], [[82.18612894790365, 60.913211494319675, 57.71421100037772, 51.41499840195253], [82.3706307929221, 69.64872010924834, 60.93790859168434, 52.98980155155883]]] 

matriz = np.array(arreglo)

print("all",matriz)
print("len",len(matriz))
print("filas",len(matriz[0]))
print("Columnas",len(matriz[0][0]))
print("Matriz0",matriz[0][1,0])
print("Matriz1",matriz[1][1,0])

df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
     index=['cobra', 'viper', 'sidewinder'],
     columns=['max_speed', 'shield'])

df =df.assign(UAR="")

resul=[]
media=[]
std=[]
for i in range(len(matriz[0])):    
    for j in range(len(matriz[0][0])):                                
        media.append("para c="+str(i)+" y para degree="+str(j)+" == "+str(np.mean([matriz[0][i, j], matriz[1][i, j] ])))
        std.append("para c="+str(i)+" y para degree="+str(j)+" == "+str(np.std([matriz[0][i, j], matriz[1][i, j] ])))

print(media)


from sklearn.preprocessing import StandardScaler
import warnings
import json
import opensmile
import pandas as pd
import numpy as np


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
data_processing = {'list':[],'model':[],'type_list':[], 'Kerner':[] , 'value_C':[], 'value_degree':[], 'Score':[], 'UAR':[] }      
modelo = SVC(kernel='poly')
matriz_uar= list()
matriz_precision = list()
matriz_f1Score=list()
matriz_recall=list()
matriz_average_precision=list()
nfold = 1
while nfold <= 2:      
    X_train, y_train = split_data("data/csv/train_phrase_both_meta_data_fold"+str(nfold)+".csv")
    X_test, y_test = split_data("data/csv/test_phrase_both_meta_data_fold"+str(nfold)+".csv")  
    c_uar = list()
    c_precision = list()
    c_f1Score=list()
    c_recall=list()
    c_average_precision=list()
    #length=range(1,64, 4)       
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
            
            # print("Precision",precision)
            # print("f1Score",f1Score)
            # print("Recall",recall)
            # print("AUC",average_precision)
            
            
            data_processing['list'].append("phrase_fold"+str(nfold)+".csv")
            data_processing['model'].append("SVC")
            data_processing['type_list'].append("Phrase")
            data_processing['Kerner'].append("poly")
            data_processing['value_C'].append(c)
            data_processing['value_degree'].append(deg)
            data_processing['Score'].append(score*100)
            data_processing['UAR'].append(uar*100)                                

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

matriz_uar = np.array(matriz_uar)
matriz_precision = np.array(matriz_precision)
matriz_f1Score = np.array(matriz_f1Score)
matriz_recall = np.array(matriz_recall)
matriz_average_precision = np.array(matriz_average_precision)  

media=[]
std=[]
for i in range(len(matriz_uar[0])):    
    for j in range(len(matriz_uar[0][0])):                                
        media.append("para c="+str(i)+" y para degree="+str(j)+" == "+str(np.mean([matriz_uar[0][i, j], matriz_uar[1][i, j] ])))
        std.append("para c="+str(i)+" y para degree="+str(j)+" == "+str(np.std([matriz_uar[0][i, j], matriz_uar[1][i, j] ])))


df = pd.DataFrame(data_processing, columns = ['list','model','type_list', 'Kerner' , 'value_C', 'value_degree', 'Score','UAR'])
df = df.assign(UAR_=media)
df = df.assign(STD_=std)
df.to_excel('data/xls/svc_phrase.xlsx', sheet_name='svc_phrase', index=False)
    