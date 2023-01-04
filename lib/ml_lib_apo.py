import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

from sklearn.impute import KNNImputer
from sklearn.dummy import DummyClassifier

## Función para evaluar cada modelo

def evaluar_modelo(modelo, X_train, y_train, X_test, y_test, plot_matrix=True, plot_roc_auc=True):
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    acc = accuracy_score(y_test, predicciones)
    precision = precision_score(y_test, predicciones, zero_division=True)
    recall = recall_score(y_test, predicciones, zero_division=True)
    f1s = f1_score(y_test, predicciones, zero_division=True)
    print('Accuracy = {0}\nPrecision = {1}\nRecall = {2}\nF1 = {2}'.format(acc, precision, recall, f1s))
    if plot_matrix:
        conf_mx = confusion_matrix(
            y_test,
            predicciones
        )
        plt.figure(figsize=(15,10))
        plot_mc = ConfusionMatrixDisplay(conf_mx)
        plot_mc.plot()
    if plot_roc_auc:
        RocCurveDisplay.from_estimator(modelo, X_test, y_test)
        #plt.show()
    return modelo


# Función para procesar resultados de GridSearchCV

def presentar_resultados (grid, orden='precision'):
    ranking = 'rank_test_' + orden
    df = pd.DataFrame(grid.cv_results_)
    columnas = [ x for x in df.columns if str(x).endswith(orden) ]
    # print(df[columnas].sort_values(by=ranking, ascending=True).head())

    print('*** Mejores resultado *** ')
    print(grid.best_score_)
    print('*** Mejores parámetros *** ')
    print(grid.best_params_)
    return df[columnas].sort_values(by=ranking, ascending=True).head()


# Funcion para evaluar según resultados

def evaluar_resultados (y_verdaderos, y_predicciones):
    a = accuracy_score(y_verdaderos, y_predicciones)
    p = precision_score(y_verdaderos, y_predicciones)
    r = recall_score(y_verdaderos, y_predicciones)
    print('Accuracy(Exactitud)={0}\nPrecision={1}\nRecall(exhaustividad)={2}'.format(a, p, r))
    

# Devuelve preprocesador sea StandardScaler o MinMaxScaler

def preparar_preprocesador(feat_num, feat_cat, tipo='standard', imputer_strategy='most_frequent'):
    if imputer_strategy == 'knn':
        imputer = ('knn_imputer', KNNImputer(n_neighbors=10, weights='distance'))
    else:
        # ['mean', 'median', 'most_frequent', 'constant']
        imputer = ('simple_imputer', SimpleImputer(strategy=imputer_strategy))

    if tipo=='minmax':
        scaler = ('minmax_scaler', MinMaxScaler())
    else:
        scaler = ('standard_scaler', StandardScaler(with_mean=True, with_std=True))

    pipeline_numerico = Pipeline(steps=[
        imputer,
        scaler
    ])
    pipeline_categorico = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(transformers=[
        ('num_pipe', pipeline_numerico, feat_num),
        ('cat_pipe', pipeline_categorico, feat_cat)
    ])
    return preprocessor
