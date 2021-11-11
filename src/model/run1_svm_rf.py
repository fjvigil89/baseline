# Tratamiento de datos
# ==============================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

# Preprocesado y modelado
# ==============================================================================
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
warnings.filterwarnings('ignore')

path_test = "../data/lst/test_phrase_both_meta_data_fold1.json"
path_train = "../data/lst/train_phrase_both_meta_data_fold1.json"
label = pd.read_csv(path_test)

print(label.head(3))
# print("Load the data model")

# data = data.drop(columns=['start', 'end'])
# data.loc[data["file"].str.split(
#     "/", expand=True)[2] == "PATH", 'file'] = 'PATH'
# data.loc[data["file"].str.split(
#     "/", expand=True)[2] == "NORM", 'file'] = 'NORM'

# X = data.drop(columns=['file'], axis=1)
# y = data['file']

# X = X.astype(float).fillna(0.0)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.8, random_state=42, stratify=y)

# print(data['file'].value_counts())


# print("Standarization")
# sc = StandardScaler()
# X_train_array = sc.fit_transform(X_train.values)
# X_train = pd.DataFrame(X_train_array, index=X_train.index,
#                        columns=X_train.columns)
# X_test_array = sc.transform(X_test.values)
# X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)

# # Creación del modelo SVM
# # ==============================================================================
# print("Modelo SVM")
# modelo = SVC(C=50, kernel='poly', random_state=42)
# modelo.fit(X_train, y_train)

# print("Score in model SVM")
# print(modelo.score(X_test, y_test))

# # Creación del modelo RF
# # ==============================================================================
# print("Modelo Ramdom Forest")
# classifier = RandomForestClassifier(n_estimators=50)
# classifier = classifier.fit(X_train, y_train)

# print("Score in model Ramdom Forest")
# print(classifier.score(X_test, y_test))
