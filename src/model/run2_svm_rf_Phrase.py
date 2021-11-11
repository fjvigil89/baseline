# Tratamiento de datos
# ==============================================================================
from sklearn.metrics import average_precision_score
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
df.loc[split[2] == "PATH", 'label'] = 1
df.loc[split[2] == "NORM", 'label'] = 0


df_mask = df['type'] == 'phrase'
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
print("........phrase.........")

print("Modelo SVM")
modelo = SVC(C=2, kernel='poly', degree=2)


cv = KFold(n_splits=5, shuffle=True)
print("cross val Score in model SVM")
scores = cross_val_score(modelo, X, y, cv=cv)
print(np.mean(scores))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=42, stratify=y)

modelo.fit(X_train, y_train)
y_score = modelo.decision_function(X_test)

print(y_score)

""" average_precision = average_precision_score(y_train, y_test)
print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
 """
