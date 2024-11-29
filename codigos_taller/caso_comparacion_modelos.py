import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils import resample

# Función para preparar los datos
def preparar_datos(df, columnas_x, columna_y):
    X = df[columnas_x]
    y = df[columna_y]
    columnas_numericas = X.select_dtypes(include=['int64', 'float64']).columns
    columnas_categoricas = X.select_dtypes(include=['object', 'category']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columnas_numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas)
        ])
    return X, y, preprocessor
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np


# Función para calcular las métricas de fairness
def calcular_metricas_fairness(y_true, y_pred, y_score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tasa_seleccion = np.mean(y_pred == 1)
    metricas = {'accuracy': accuracy, 'fpr': fpr, 'tpr': tpr, 'tasa_seleccion': tasa_seleccion}
    if y_score is not None:
        try:
            metricas['auc'] = roc_auc_score(y_true, y_score)
        except:
            metricas['auc'] = None
    return metricas

# Función para calcular disparidad en FPR (solo FPR se considera disparidad)
def calcular_disparidad_fpr(valores_grupos):
    FPR_max = max(valores_grupos)
    FPR_min = min(valores_grupos)
    disparidad = abs(FPR_max - FPR_min)  # Disparidad en FPR (diferencia absoluta)
    return disparidad

# Función para simular y calcular disparidades globales
def simular_modelos_comparacion(df_original, columnas_x, columna_y, columna_sensible='Genero', num_iteraciones=100):
    modelos = {
        'base': {'columnas': columnas_x},
        'sin_genero': {'columnas': [col for col in columnas_x if col != columna_sensible]},
        'atenuado': {'columnas': columnas_x}  # El modelo atenuado se considera con regularización L2
    }
    resultados = {
        modelo_nombre: {
            grupo: {
                'metricas': []
            } for grupo in df_original[columna_sensible].unique()
        } for modelo_nombre in modelos.keys()
    }
    grupos_unicos = df_original[columna_sensible].unique()
    
    for i in range(num_iteraciones):
        df_bootstrap = resample(df_original, replace=True, n_samples=len(df_original), random_state=i*2)
        
        for modelo_nombre, config in modelos.items():
            X, y, preprocessor = preparar_datos(df_bootstrap, config['columnas'], columna_y)
            
            # Para el modelo "atenuado", agregamos regularización L2 en LogisticRegression
            if modelo_nombre == 'atenuado':
                modelo = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', LogisticRegression(random_state=42, class_weight='balanced', penalty='l2', C=0.1))  # Regularización L2
                ])
            else:
                modelo = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', LogisticRegression(random_state=42))
                ])
            
            modelo.fit(X, y)
            
            for grupo in grupos_unicos:
                df_grupo = df_bootstrap[df_bootstrap[columna_sensible] == grupo]
                X_grupo, y_grupo, _ = preparar_datos(df_grupo, config['columnas'], columna_y)
                predicciones_grupo = modelo.predict(X_grupo)
                try:
                    probabilidades_grupo = modelo.predict_proba(X_grupo)[:, 1]
                except:
                    probabilidades_grupo = None
                metricas = calcular_metricas_fairness(y_grupo, predicciones_grupo, probabilidades_grupo)
                resultados[modelo_nombre][grupo]['metricas'].append(metricas)
    
    resultados_procesados = {modelo: {} for modelo in modelos.keys()}
    
    for modelo_nombre in modelos.keys():
        for grupo, datos_grupo in resultados[modelo_nombre].items():
            metricas_iteraciones = datos_grupo['metricas']
            metricas_promedio = {f'{metrica}_promedio': np.mean([m.get(metrica, 0) for m in metricas_iteraciones])
                                 for metrica in ['accuracy', 'fpr', 'tpr', 'tasa_seleccion', 'auc']}
            resultados_procesados[modelo_nombre][grupo] = metricas_promedio
        
        # Cálculo de disparidad solo para FPR
        disparidades = {}
        fpr_valores = [
            resultados_procesados[modelo_nombre][grupo].get('fpr_promedio', 0)
            for grupo in grupos_unicos
        ]
        disparidad_fpr = calcular_disparidad_fpr(fpr_valores)
        disparidades['disparidad_fpr'] = disparidad_fpr
        
        # Calcular diferencias en accuracy y tasa de selección
        diferencias = {}
        for metrica in ['accuracy', 'tasa_seleccion']:
            valores_grupos = [
                resultados_procesados[modelo_nombre][grupo].get(f'{metrica}_promedio', 0)
                for grupo in grupos_unicos
            ]
            diferencia = max(valores_grupos) - min(valores_grupos)
            diferencias[f'diferencia_{metrica}'] = diferencia
        
        resultados_procesados[modelo_nombre]['disparidades_globales'] = disparidades
        resultados_procesados[modelo_nombre]['diferencias_globales'] = diferencias
    
    return resultados_procesados


# Función para mostrar resultados en formato texto
def mostrar_resultados_texto(resultados):
    for modelo, valores in resultados.items():
        print(f"Modelo: {modelo}")
        for grupo, metrics in valores.items():
            if grupo != 'disparidades_globales':
                print(f"  {grupo}:")
                for metric, valor in metrics.items():
                    print(f"    {metric}: {valor}")
        print("\nDisparidades Globales:")
        for metric, valor in valores['disparidades_globales'].items():
            print(f"  {metric}: {valor}")
        print("\n" + "-"*50)


# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos
    df = pd.read_csv("simulacion_contratacion_tech_chile.csv")

    # Definir columnas
    columnas_x = ['Edad', 'Genero', 'NivelEducativo', 'RamaPrincipal', 'AñosExperiencia',
                  'AñosExperienciaPro', 'HabilidadesComputacionales', 'SalarioAnterior']
    columna_y = "Contratacion"

      # Ejecutar simulación

    print("\nEjecutando simulación de modelos...")
    resultados = simular_modelos_comparacion(
        df_original=df,
        columnas_x=columnas_x,
        columna_y=columna_y,
        columna_sensible='Genero',
        num_iteraciones=50
    )
            
            
        # Mostrar resultados detallados
    resultados_completos = {}
    for modelo, datos_modelo in resultados.items():
        modelo_resultado = {}
        for metrica, valores_metrica in datos_modelo.items():
            if isinstance(valores_metrica, dict):
                metrica_detalle = {grupo: valor for grupo, valor in valores_metrica.items()}
            else:
                metrica_detalle = valores_metrica
            modelo_resultado[metrica] = metrica_detalle
        resultados_completos[modelo] = modelo_resultado

    mostrar_resultados_texto(resultados_completos) 

