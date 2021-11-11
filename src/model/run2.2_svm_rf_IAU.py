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
from sklearn.model_selection import cross_val_score, KFold
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

path = "./data/csv/_smile.csv"
df = pd.read_csv(path)


print("Load the data model")


df = df.assign(type="")
df = df.assign(label="")

split = df["file"].str.split(r"\/|-", expand=True)

df['type'] = 'IAU'
df.loc[split[5] == "phrase.wav", 'type'] = 'phrase'
df.loc[split[5] == "a_n.wav", 'type'] = 'vowel'
df.loc[split[5] == "i_n.wav", 'type'] = 'vowel'
df.loc[split[5] == "u_n.wav", 'type'] = 'vowel'
df.loc[split[2] == "PATH", 'label'] = 'PATH'
df.loc[split[2] == "NORM", 'label'] = 'NORM'

# Filtrar por vowel

df_mask = df['type'] == 'IAU'
filtered_df = df[df_mask]

print(filtered_df['type'].value_counts())

data = filtered_df.drop(columns=['file', 'start', 'end', 'type'])

#print(data["file"].str.split(r"\/|-", expand=True))
# print(data.head(3))


print("all PATH or NORM")
X = data.drop(columns=['label'], axis=1)
y = data['label']

X = X.astype(float).fillna(0.0)

print(data['label'].value_counts())


print("Standarization")
sc = StandardScaler()
X_array = sc.fit_transform(X.values)
X = pd.DataFrame(X_array, index=X.index,
                 columns=X.columns)

# Creación del modelo SVM
# ==============================================================================
print("........IAU.........")

print("Modelo SVM")
modelo = SVC(C=50, kernel='poly', degree=1)


cv = KFold(n_splits=5, shuffle=True)
print("cross val Score in model SVM")
scores = cross_val_score(modelo, X, y, cv=cv)
print(np.mean(scores))
