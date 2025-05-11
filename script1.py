import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import imblearn

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import *
from sklearn.decomposition import *
from sklearn.feature_selection import *
from sklearn.model_selection import *
from sklearn.metrics import *
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from tqdm import tqdm
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone


### MEJORES HIPERPARAMETROS DE LOS MODELOS ###

best_knn = KNeighborsRegressor(
    weights='distance',
    n_neighbors=int(np.int64(16)),
    metric='manhattan'
)
best_rf = RandomForestRegressor(
    n_estimators=200,
    min_samples_split=5,
    min_samples_leaf=1,
    max_depth=20,
    bootstrap=True,random_state=42
)
best_hgb = HistGradientBoostingRegressor(
    min_samples_leaf=50,
    max_iter=300,
    max_depth=5,
    learning_rate=0.1,
    l2_regularization=0.1,random_state=42
)
best_cat = CatBoostRegressor(
    learning_rate=0.05,
    l2_leaf_reg=5,
    iterations=1000,
    depth=6,
    verbose=0,random_state=42
)
best_lgbm = LGBMRegressor(
    subsample=0.8,
    num_leaves=50,
    n_estimators=300,
    min_child_samples=30,
    max_depth=6,
    learning_rate=0.05,
    colsample_bytree=0.6,random_state=42
)
best_xgb = XGBRegressor(
    subsample=1.0,
    reg_lambda=1.0,
    reg_alpha=1.0,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    colsample_bytree=0.8,random_state=42
)
best_mlp = MLPRegressor(
    solver='adam',
    max_iter=1000,
    learning_rate_init=0.01,
    hidden_layer_sizes=(64, 32),
    activation='relu',random_state=42
)

best_et = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    bootstrap=False,
    random_state=42
)

best_ridge = Ridge(alpha=0.1, solver='auto', random_state=42)

best_svr = SVR(kernel='linear', gamma='scale', epsilon=0.01, C=100)

best_knn2 = KNeighborsRegressor(
    weights='distance',
    n_neighbors=int(np.int64(16)),
    metric='manhattan'
)
best_rf2 = RandomForestRegressor(
    n_estimators=200,
    min_samples_split=5,
    min_samples_leaf=1,
    max_depth=20,
    bootstrap=True,random_state=15
)
best_hgb2 = HistGradientBoostingRegressor(
    min_samples_leaf=50,
    max_iter=300,
    max_depth=5,
    learning_rate=0.1,
    l2_regularization=0.1,random_state=15
)
best_cat2 = CatBoostRegressor(
    learning_rate=0.05,
    l2_leaf_reg=5,
    iterations=1000,
    depth=6,
    verbose=0,random_state=15
)
best_lgbm2 = LGBMRegressor(
    subsample=0.8,
    num_leaves=50,
    n_estimators=300,
    min_child_samples=30,
    max_depth=6,
    learning_rate=0.05,
    colsample_bytree=0.6,random_state=15
)
best_xgb2 = XGBRegressor(
    subsample=1.0,
    reg_lambda=1.0,
    reg_alpha=1.0,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    colsample_bytree=0.8,random_state=15
)
best_mlp2 = MLPRegressor(
    solver='adam',
    max_iter=1000,
    learning_rate_init=0.01,
    hidden_layer_sizes=(64, 32),
    activation='relu',random_state=15
)
best_et2 = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    bootstrap=False,
    random_state=15
)

best_ridge2 = Ridge(alpha=0.1, solver='auto', random_state=15)


#### ENTRENANDO CON EL DATASET DE TRAIN COMPLETO ###

#  1. Cargar dataset de entrenamiento 
df = pd.read_csv('./data/train.csv', delimiter=',')

#  2. Guardar y separar target 
target = df['prezo_euros']
X = df.drop(columns=['id', 'prezo_euros'])

#  3. Codificar variables categóricas 
# Label Encoding para eficiencia_enerxetica
le = LabelEncoder()
X['eficiencia_enerxetica'] = le.fit_transform(X['eficiencia_enerxetica'])

# One-Hot Encoding para el resto
X = pd.get_dummies(X, columns=[
    'tipo_edificacion',
    'calidade_materiais',
    'cor_favorita_propietario',
    'acceso_transporte_publico',
    'orientacion'
])

#  4. Rellenar NaNs con mediana 
X = X.fillna(X.median(numeric_only=True))

#  5. Separar en train y test
X_train_full = X.copy()
y_train_full = target.copy()

#  6. Normalizar 
#  6.1 Primero setear el scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_train_full)


########################################################################################
########################################################################################


#Indicamos cuales van a ser los modelos iniciales a entrenar
modelos_iniciales = [
     best_knn,
    best_rf,
    best_hgb,
    best_cat, 
    best_lgbm, 
    best_xgb,
    best_svr,
    best_et,
    best_ridge,
    best_knn2,
    best_rf2,
    best_hgb2,
    best_cat2,
    best_lgbm2,
    best_xgb2,
    best_et2,
    best_ridge2
]

#Indicamos cual va a ser el metamodelo a usar
metamodelo = StackingRegressor(
    estimators=[
        ('lr', LinearRegression()),
        ('mlp_deep', MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', max_iter=3000, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
        ('mlpo',MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',solver='adam', max_iter=3000, early_stopping=True,random_state=42))
    ],
    final_estimator=LinearRegression(),  # Meta-modelo final
    cv=5,  # Validación cruzada
    n_jobs=-1
)

X_meta_train = np.zeros((len(X_train_scaled), len(modelos_iniciales) + X_train_scaled.shape[1]))
X_meta_test = np.zeros((len(X_test_scaled), len(modelos_iniciales) + X_test_scaled.shape[1]))

splits=5
kf = KFold(n_splits=splits, shuffle=True, random_state=42)
modelos_entrenados_finales = []  # Para guardar modelos entrenados finales

for modelo_idx, modelo in enumerate(tqdm(modelos_iniciales, desc="Entrenando modelos...")):
    nombremodel = modelo.__class__.__name__
    tqdm.write(f"Entrenando: {nombremodel}")
    
    predicciones_ext = [None] * len(X_train_scaled)
    predicciones_int = []

    for train_idx, val_idx in kf.split(X_train_scaled):
        X_train_int, y_train_int = X_train_scaled[train_idx], y_train_full.iloc[train_idx]
        X_val = X_train_scaled[val_idx]

        modelo_cv = clone(modelo)
        modelo_cv.fit(X_train_int, y_train_int)

        preds_val = modelo_cv.predict(X_val)
        for idx, pred in zip(val_idx, preds_val):
            predicciones_ext[idx] = pred

        predicciones_int.append(modelo_cv.predict(X_test_scaled))

    for i, pred in enumerate(predicciones_ext):
        X_meta_train[i, modelo_idx] = pred

    avg_test_preds = np.mean(predicciones_int, axis=0)
    for i, pred in enumerate(avg_test_preds):
        X_meta_test[i, modelo_idx] = pred

    modelo_entrenado = clone(modelo)
    modelo_entrenado.fit(X_train_scaled, y_train_full)
    modelos_entrenados_finales.append(modelo_entrenado)

X_meta_train[:, len(modelos_iniciales):] = X_train_scaled
X_meta_test[:, len(modelos_iniciales):] = X_test_scaled

metamodelo.fit(X_meta_train, y_train_full)
y_pred_provisional = metamodelo.predict(X_meta_test)

#  9. Cargar test.csv 
df_test = pd.read_csv('./data/test.csv', delimiter=',')

# Guardar el ID
id_test = df_test['id']
df_test = df_test.drop(columns=['id'])

#  10. Codificar igual que en entrenamiento 
df_test['eficiencia_enerxetica'] = le.transform(df_test['eficiencia_enerxetica'])

df_test = pd.get_dummies(df_test, columns=[
    'tipo_edificacion',
    'calidade_materiais',
    'cor_favorita_propietario',
    'acceso_transporte_publico',
    'orientacion'
])

# Alinear columnas con X_train
df_test = df_test.reindex(columns=X_train_full.columns, fill_value=0)

# Rellenar NaNs
df_test = df_test.fillna(X_train_full.median(numeric_only=True))

#  11. Normalizar 
X_test_final = scaler.transform(df_test)

predicciones_reales=[]
for modelo in modelos_entrenados_finales:
    predicciones_reales.append(modelo.predict(X_test_final))

stack_predicciones=np.column_stack(predicciones_reales)

#  12. Predecir 
y_pred_final = metamodelo.predict(np.hstack((stack_predicciones, X_test_final)))

#  13. Guardar resultados 
resultados = pd.DataFrame({
    'id': id_test,
    'prezo_euros': y_pred_final
})
print('PREDICCIONES GENERADAS')
resultados.to_csv('predicciones_blending_doble_comprobar.csv', index=False)