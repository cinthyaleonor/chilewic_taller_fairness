import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    classification_report, 
    roc_auc_score
)
from sklearn.utils import resample

def calcular_metricas_fairness(y_true, y_pred, y_score=None):
    """
    Calcula métricas de fairness más comprehensivas.
    
    Parámetros:
    - y_true: Etiquetas verdaderas
    - y_pred: Predicciones del modelo
    - y_score: Probabilidades de predicción (opcional)
    
    Retorna:
    - Diccionario con métricas de fairness
    """
    # Calcular matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Métricas base
    accuracy = accuracy_score(y_true, y_pred)
    
    # Tasas de error
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Tasas de verdaderos positivos y negativos
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Tasa de selección (proporción de predicciones positivas)
    tasa_seleccion = np.mean(y_pred == 1)
    
    # Métricas adicionales de fairness
    metricas = {
        'accuracy': accuracy,
        'fpr': fpr,
        'fnr': fnr,
        'tpr': tpr,
        'tnr': tnr,
        'tasa_seleccion': tasa_seleccion
    }
    
    # Añadir AUC-ROC si se proporcionan probabilidades
    if y_score is not None:
        try:
            metricas['auc'] = roc_auc_score(y_true, y_score)
        except:
            metricas['auc'] = None
    
    return metricas

def simular_modelos_comparacion(df_original, columnas_x, columna_y, columna_sensible='Genero', num_iteraciones=100):
    """
    Simula y compara modelos para análisis de fairness.
    
    Parámetros:
    - df_original: DataFrame original
    - columnas_x: Características para el modelo
    - columna_y: Variable objetivo
    - columna_sensible: Columna para análisis de fairness
    - num_iteraciones: Número de iteraciones de bootstrap
    
    Retorna:
    - Resultados detallados de fairness por modelo y grupo sensible
    """
    # Configuración de los modelos
    modelos = {
        'base': {'columnas': columnas_x},
        'sin_genero': {'columnas': [col for col in columnas_x if col != columna_sensible]},
        'atenuado': {'columnas': columnas_x}
    }
    
    # Estructura para almacenar resultados
    resultados = {
        modelo_nombre: {
            grupo: {
                'metricas': [],
                'disparidad_metrica': {}
            } for grupo in df_original[columna_sensible].unique()
        } for modelo_nombre in modelos.keys()
    }

    # Obtener grupos únicos
    grupos_unicos = df_original[columna_sensible].unique()

    # Iterar sobre bootstrap
    for i in range(num_iteraciones):
        # Generar muestra bootstrap
        df_bootstrap = resample(df_original, replace=True, n_samples=len(df_original), random_state=i*2)
        
        # Entrenar y evaluar cada modelo
        for modelo_nombre, config in modelos.items():
            # Preparar datos
            X, y, preprocessor = preparar_datos(df_bootstrap, config['columnas'], columna_y)
            
            # Configurar modelo
            if modelo_nombre == 'atenuado':
                modelo = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', LogisticRegression(
                        random_state=42,
                        class_weight='balanced',
                        penalty='l1',
                        solver='liblinear'
                    ))
                ])
            else:
                modelo = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', LogisticRegression(random_state=42))
                ])
            
            # Ajustar modelo
            modelo.fit(X, y)
            
            # Calcular métricas por grupo sensible
            for grupo in grupos_unicos:
                # Filtrar datos por grupo
                df_grupo = df_bootstrap[df_bootstrap[columna_sensible] == grupo]
                X_grupo, y_grupo, _ = preparar_datos(df_grupo, config['columnas'], columna_y)
                
                # Predecir
                predicciones_grupo = modelo.predict(X_grupo)
                
                # Obtener probabilidades si está disponible
                try:
                    probabilidades_grupo = modelo.predict_proba(X_grupo)[:, 1]
                except:
                    probabilidades_grupo = None
                
                # Calcular métricas para este grupo
                metricas = calcular_metricas_fairness(y_grupo, predicciones_grupo, probabilidades_grupo)
                resultados[modelo_nombre][grupo]['metricas'].append(metricas)

    # Procesar resultados
    resultados_procesados = {modelo: {} for modelo in modelos.keys()}
    
    for modelo_nombre in modelos.keys():
        # Obtener grupos únicos
        for grupo, datos_grupo in resultados[modelo_nombre].items():
            # Extraer métricas de todas las iteraciones
            metricas_iteraciones = datos_grupo['metricas']
            
            # Calcular promedios y desviaciones
            metricas_promedio = {}
            for metrica in ['accuracy', 'fpr', 'fnr', 'tpr', 'tnr', 'tasa_seleccion', 'auc']:
                valores = [m.get(metrica, 0) for m in metricas_iteraciones if m.get(metrica) is not None]
                
                if valores:
                    metricas_promedio[f'{metrica}_promedio'] = np.mean(valores)
                    metricas_promedio[f'{metrica}_std'] = np.std(valores)
            
            resultados_procesados[modelo_nombre][grupo] = metricas_promedio

        # Calcular disparidades entre grupos
        disparidades = {}
        for metrica in ['accuracy', 'fpr', 'tpr']:
            # Obtener los promedios de la métrica para cada grupo
            valores_grupos = [
                resultados_procesados[modelo_nombre][grupo].get(f'{metrica}_promedio', 0) 
                for grupo in grupos_unicos
            ]
            
            # Calcular la disparidad (diferencia máxima entre grupos)
            if len(valores_grupos) > 1:
                disparidad = max(valores_grupos) - min(valores_grupos)
                disparidades[f'disparidad_{metrica}'] = disparidad
        
        # Agregar disparidades a los resultados de cada grupo
        for grupo in grupos_unicos:
            resultados_procesados[modelo_nombre][grupo].update(disparidades)

    return resultados_procesados

def preparar_datos(df, columnas_x, columna_y):
    """
    Prepara los datos para el modelo de machine learning.
    
    Parámetros:
    - df: DataFrame con los datos
    - columnas_x: Lista de características para el modelo
    - columna_y: Variable objetivo
    
    Retorna:
    - X: Características preprocesadas
    - y: Variable objetivo
    - preprocessor: Transformador de características
    """
    X = df[columnas_x]
    y = df[columna_y]
    
    # Identificar columnas numéricas y categóricas
    columnas_numericas = X.select_dtypes(include=['int64', 'float64']).columns
    columnas_categoricas = X.select_dtypes(include=['object', 'category']).columns
    
    # Crear preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columnas_numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas)
        ])
    
    return X, y, preprocessor

def crear_tabla_comparativa(resultados):
    """
    Crea una tabla comparativa de los resultados de fairness.
    
    Parámetros:
    - resultados: Resultados procesados de la simulación
    
    Retorna:
    - Tabla de resultados formateada
    """
    tabla = {}
    
    for modelo_nombre, datos_modelo in resultados.items():
        tabla[modelo_nombre] = {}
        
        # Métricas a extraer con sus sufijos
        metricas_base = {
            'accuracy': ['_promedio', '_std'],
            'fpr': ['_promedio', '_std'],
            'tpr': ['_promedio', '_std'],
            'tasa_seleccion': ['_promedio', '_std'],
            'auc': ['_promedio', '_std'],
            'disparidad_accuracy': [],
            'disparidad_fpr': [],
            'disparidad_tpr': []
        }
        
        for metrica_base, sufijos in metricas_base.items():
            if not sufijos:
                # Para métricas globales sin sufijos (las disparidades)
                # Tomamos el valor de cualquier grupo ya que es el mismo para todos
                primer_grupo = list(datos_modelo.keys())[0]
                tabla[modelo_nombre][metrica_base] = f"{datos_modelo[primer_grupo].get(metrica_base, 0):.3f}"
            else:
                # Para métricas con promedio y desviación
                tabla[modelo_nombre][metrica_base] = {}
                for grupo, datos_grupo in datos_modelo.items():
                    valores = {}
                    for sufijo in sufijos:
                        clave_completa = f"{metrica_base}{sufijo}"
                        valores[sufijo] = datos_grupo.get(clave_completa, 0)
                    
                    # Formatear según el sufijo
                    if '_promedio' in valores:
                        tabla[modelo_nombre][metrica_base][grupo] = f"{valores.get('_promedio', 0):.3f} ± {valores.get('_std', 0):.3f}"
    
    return tabla

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
    
    # Crear tabla comparativa
    print("\nCreando tabla comparativa...")
    tabla_resultados = crear_tabla_comparativa(resultados)
    
    # Mostrar resultados detallados
    print("\nResultados por modelo:")
    for modelo, datos_modelo in tabla_resultados.items():
        print(f"\n{modelo.upper()}:")
        for metrica, valores_metrica in datos_modelo.items():
            print(f"\n{metrica}:")
            # Verificar si valores_metrica es un diccionario o un valor único
            if isinstance(valores_metrica, dict):
                for grupo, valor in valores_metrica.items():
                    print(f"{grupo}: {valor}")
            else:
                # Si es un valor único (como las disparidades)
                print(f"Global: {valores_metrica}")