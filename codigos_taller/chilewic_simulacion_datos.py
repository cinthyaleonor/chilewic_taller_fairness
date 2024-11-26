import numpy as np
import pandas as pd
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix




def generar_dataset(num_samples=1000, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
    # Género con sesgo en probabilidades (incluye 'No Binario')
    genero = np.random.choice(['Hombre', 'Mujer', 'No Binario'], 
                              size=num_samples, 
                              p=[0.65, 0.3, 0.05])

    # Nivel Educativo con sesgo según género
    def generar_nivel_educativo(genero):
        if genero == 'Hombre':
            return np.random.choice(['Técnico', 'Universitario', 'Magister', 'Doctorado'],
                                    p=[0.4, 0.45, 0.10, 0.05])
        elif genero == 'Mujer':
            return np.random.choice(['Técnico', 'Universitario', 'Magister', 'Doctorado'],
                                    p=[0.4, 0.5, 0.07, 0.03])
        else:  # No Binario
            return np.random.choice(['Técnico', 'Universitario', 'Magister', 'Doctorado'],
                                    p=[0.35, 0.55, 0.05, 0.05])
    
    nivel_educativo = np.array([generar_nivel_educativo(g) for g in genero])
    
    # Rama Principal (Tecnología)
    ramas_principales = ['Desarrollo', 'Infraestructura', 'Ciberseguridad', 'Soporte', 'Análisis de Datos']
    rama_principal = np.random.choice(ramas_principales, size=num_samples, p=[0.4, 0.2, 0.1, 0.25, 0.05])

    # Edad

    def generar_edad_por_genero(genero, media_edad=35, desviacion_edad=5):
        """
        Genera una edad basada en el género con una distribución normal, pero ajustada por género.
        
        Parameters:
            genero (str): El género del individuo ('Hombre', 'Mujer', 'No Binario').
            media_edad (float): La media de la edad para cada género.
            desviacion_edad (float): La desviación estándar de la edad para cada género.
        
        Returns:
            int: La edad generada.
        """
        if genero == 'Hombre':
            return int(np.clip(np.random.normal(loc=media_edad, scale=desviacion_edad), 18, 65))
        elif genero == 'Mujer':
            return int(np.clip(np.random.normal(loc=media_edad - 3, scale=desviacion_edad), 18, 60))
        else:  # No Binario
            return int(np.clip(np.random.normal(loc=media_edad - 5, scale=desviacion_edad), 18, 65))

    
    # Generar edades para cada género
    edad = np.array([generar_edad_por_genero(g) for g in genero])
           
    
        # Función para generar años de experiencia basada en la edad
    def generar_anns_experiencia(edad):
        """
        Genera los años de experiencia basados en la edad.
        La experiencia aumenta con la edad, con algo de variabilidad.
        """
        # La experiencia es una fracción de la edad con algo de variabilidad
        experiencia_base = int(edad * 0.3)  # Años de experiencia son el 30% de la edad (ajustable)
        # Agregar ruido para hacerlo más realista (puede ser ajustado)
        experiencia_con_ruido = experiencia_base + np.random.randint(-3, 4)  # Rango de variabilidad entre -3 y 3
        return max(experiencia_con_ruido, 0)  # No permitir que la experiencia sea negativa
    
    # Generar años de experiencia basados en la edad
    anns_experiencia_base = np.array([generar_anns_experiencia(e) for e in edad])

    
    # Ajuste de años de experiencia según nivel educativo
    def ajustar_experiencia(exp_base, nivel):
        ajustes = {
            'Técnico': 0,
            'Universitario': 0,
            'Magister': 2,
            'Doctorado': 4
        }
        return exp_base + ajustes[nivel]
    
    anns_experiencia = np.array([
        min(ajustar_experiencia(exp, nivel), edad[i] - 18)
        for i, (exp, nivel) in enumerate(zip(anns_experiencia_base, nivel_educativo))
    ])

    # Años programando (no puede ser mayor que los años de experiencia)
    anns_programando = np.minimum(anns_experiencia, np.random.normal(6, 4, size=num_samples).clip(0, 25).astype(int))
       
    # Tecnologías con las que ha trabajado
    tecnologias_disponibles = ['Python', 'JavaScript', 'Java', 'C++', 'SQL', 'AWS', 'R', 'Docker', 'Kubernetes', 'React', 'Angular', 'Node.js']
    tecnologias_trabajadas = [random.sample(tecnologias_disponibles, random.randint(1, 4)) 
        for _ in range(num_samples)]
    
    # Asignar puntajes aleatorios (de 1 a 10) a cada tecnología trabajada
    puntajes_tecnologia = [
        {tec: random.randint(1, 10) for tec in tech_list} 
        for tech_list in tecnologias_trabajadas
    ]

    def calcular_habilidad_computacional(anns_programando, tecnologias_trabajadas, puntajes_tecnologia, nivel_educativo):
        factor_experiencia = 1.2
        factor_educativo = {
            'Técnico': 0.8,
            'Universitario': 1.0,
            'Magister': 1.1,
            'Doctorado': 1.2
        }
        
        habilidades_computacionales = [
            (sum(puntajes_tecnologia[i].values()) + (anns_programando[i] * factor_experiencia)) * 
            factor_educativo[nivel_educativo[i]]
            for i in range(len(tecnologias_trabajadas))
        ]
        return habilidades_computacionales

    def normalizar_habilidades(habilidades_computacionales):
        habilidades_computacionales = np.array(habilidades_computacionales)
        habilidad_min = habilidades_computacionales.min()
        habilidad_max = habilidades_computacionales.max()
        habilidades_normalizadas = 1 + 9 * (habilidades_computacionales - habilidad_min) / (habilidad_max - habilidad_min)
        habilidades_normalizadas_redondeadas = [math.floor(hab * 10) / 10 for hab in habilidades_normalizadas]
        return habilidades_normalizadas_redondeadas

    # Calcular las habilidades computacionales
    habilidades_computacionales = calcular_habilidad_computacional(
        anns_programando, tecnologias_trabajadas, puntajes_tecnologia, nivel_educativo
    )

    # Normalizar las habilidades
    habilidades_normalizadas = normalizar_habilidades(habilidades_computacionales)

    def calcular_salario(educacion, experiencia, habilidades, genero):
        # Salario base ajustado al mercado chileno (en pesos chilenos)
        salario_base = 410000  # 0.8M CLP
        factor_educativo = {
            'Técnico': 1.2, 
            'Universitario': 1.5, 
            'Magister': 1.8, 
            'Doctorado': 2.2
        }
        base = salario_base * (1 + experiencia * 0.05) * factor_educativo[educacion] * ( 1 + habilidades/10)
        if genero in ['Mujer', 'No Binario']:  # Sesgo salarial Género
            return base * 0.73
        return base
    
    salario_anterior = np.array([calcular_salario(edu, exp, prog, gen) 
                                for edu, exp, prog, gen in zip(nivel_educativo, anns_experiencia, habilidades_computacionales, genero)])
    
    #### Contratación 
    def calcular_factor_mercado(rama):
        """Calcula un factor de demanda según la rama tecnológica"""
        factores_mercado = {
            'Desarrollo': 1.2,        # Alta demanda
            'Infraestructura': 1.1,   # Demanda moderada-alta
            'Ciberseguridad': 1.3,    # Muy alta demanda
            'Soporte': 1.0,           # Demanda moderada
            'Análisis de Datos': 1.25  # Demanda alta
        }
        return factores_mercado.get(rama, 1.0)

    def calcular_factor_tecnologias(tecnologias, puntajes):
        """Calcula un factor basado en las tecnologías y sus puntajes"""
        tecnologias_demandadas = {
            'Python': 1.2,
            'JavaScript': 1.15,
            'AWS': 1.25,
            'Docker': 1.2,
            'Kubernetes': 1.3,
            'React': 1.15,
            'Node.js': 1.1
        }
        
        factor = 1.0
        for tech in tecnologias:
            if tech in tecnologias_demandadas:
                # Considera tanto la presencia de la tecnología como el puntaje
                factor *= 1 + (tecnologias_demandadas[tech] - 1) * (puntajes[tech] / 10)
        
        return min(factor, 2.0)  # Limitar el factor máximo

    def calcular_probabilidad_base(experiencia, habilidades, nivel_educativo):
        """Calcula la probabilidad base de contratación"""
        factor_educativo = {
            'Técnico': 0.6,
            'Universitario': 0.7,
            'Magister': 0.8,
            'Doctorado': 0.8
        }
        
        # Experiencia tiene un efecto logarítmico (rendimientos decrecientes)
        factor_experiencia = np.log1p(experiencia) / 3
        
        # Habilidades tienen un efecto más directo
        factor_habilidades = habilidades / 2
        
        prob_base = (factor_experiencia + factor_habilidades) * factor_educativo[nivel_educativo]
        return prob_base

    def contratacion_sigmoide(x):
        """Función sigmoide que convierte un valor en una probabilidad entre 0 y 1, 
        y luego ajusta a 1 si la probabilidad es mayor o igual a 0.5, y 0 si es menor."""
        probabilidad = 1 / (1 + np.exp(-x))  # Calculamos la sigmoide
        if probabilidad >= 0.85:
            return 1
        else:
            return 0

    def puntaje_contratacion(experiencia, habilidades, genero, nivel_educativo, rama, tecnologias, puntajes):
        """Función principal para calcular la probabilidad de contratación"""
        # Probabilidad base
        ruido = np.random.normal(0, 0.1)

        prob_base = calcular_probabilidad_base(experiencia, habilidades, nivel_educativo) 
        
        # Factores adicionales
        factor_mercado = calcular_factor_mercado(rama)
        factor_tech = calcular_factor_tecnologias(tecnologias, puntajes)
        
        # Factor de género (representando sesgos del mercado)
        factor_genero = 0.75 if genero in ['Mujer', 'No Binario'] else 0.95
        
        # Combinación de factores
        puntaje = prob_base * factor_mercado * factor_tech * factor_genero + ruido
        
        # Normalizar usando sigmoide
        return puntaje

    # Calcular probabilidades de contratación
    probabilidad_empleo = np.array([
        puntaje_contratacion(
            exp, hab, gen, edu, rama, techs, puntajes_tecnologia[i]
        ) for i, (exp, hab, gen, edu, rama, techs) in enumerate(
            zip(anns_experiencia, habilidades_normalizadas, genero, 
                nivel_educativo, rama_principal, tecnologias_trabajadas)
        )
    ])
    
    # Convertir lista de tecnologías a string con separador '|'
    tecnologias_str = ["|".join(techs) for techs in tecnologias_trabajadas]

    contratacion = np.array([contratacion_sigmoide(prob) for prob in probabilidad_empleo])

    puntajes_tecnologia_sum = [sum(puntajes_tecnologia[i].values()) for i in range(len(tecnologias_trabajadas))]
    
    df = pd.DataFrame({
        'Edad': edad,
        'Genero': genero,
        'NivelEducativo': nivel_educativo,
        'RamaPrincipal': rama_principal,
        'AñosExperiencia': anns_experiencia,
        'AñosExperienciaPro': anns_programando,
        'HaTrabajadoCon': tecnologias_str,  # Ahora usa el formato con separador '|'
        #'PuntajesTecnologias': puntajes_tecnologia_sum,
        'HabilidadesComputacionales': habilidades_normalizadas,
        'SalarioAnterior': salario_anterior,
        'Puntaje': probabilidad_empleo,
        'Contratacion':  contratacion 
    })
    
    return df


# Función para preparar los datos
def preparar_datos(df, columnas_x, columna_y):
    columnas_numericas = df[columnas_x].select_dtypes(include=['int64', 'float64']).columns
    columnas_categoricas = df[columnas_x].select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), columnas_numericas),
                      ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas)])
    
    X = df[columnas_x]
    y = df[columna_y]
    
    return X, y, preprocessor

def simular_bootstrap(df_original, columnas_x, columna_y, num_iteraciones=5):
    
    # Inicializamos el diccionario para guardar las tasas de contratación por género
    tasas_por_genero = {genero: {'tasas_seleccion': [], 'accuracy': [], 'fpr': [], 'tpr': []} 
                       for genero in df_original['Genero'].unique()}

    for i in range(num_iteraciones):
        # Generar muestra con bootstrap (muestreo con reemplazo)
        df_bootstrap = resample(df_original, replace=True, n_samples=len(df_original), random_state=i*2)
        
        # Preparar datos y entrenar modelo con todos los datos
        X, y, preprocessor = preparar_datos(df_bootstrap, columnas_x, columna_y)
        
        modelo = Pipeline([('preprocessor', preprocessor),
                         ('classifier', LogisticRegression(random_state=42))])
        
        modelo.fit(X, y)

        # Calcular métricas por género
        for genero in df_original['Genero'].unique():
            # Filtrar datos solo para este género
            df_genero = df_bootstrap[df_bootstrap['Genero'] == genero]
            
            # Preparar datos específicamente para este género
            X_genero, y_genero, _ = preparar_datos(df_genero, columnas_x, columna_y)
            
            # Realizar predicciones para este género
            predicciones_genero = modelo.predict(X_genero)
            
            # Calcular tasa de selección
            tasa_seleccion = float(sum(predicciones_genero == 1) / len(predicciones_genero))
            tasas_por_genero[genero]['tasas_seleccion'].append(tasa_seleccion)

            # Calcular matriz de confusión para este género
            if len(np.unique(y_genero)) > 1:
                tn, fp, fn, tp = confusion_matrix(y_genero, predicciones_genero).ravel()
                
                # Calcular métricas específicas para este género
                total = tp + tn + fp + fn
                accuracy = float((tp + tn) / total) if total > 0 else 0
                fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
                tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0  # TPR en vez de FNR
            else:
                accuracy = 0
                fpr = 0
                tpr = 0

            # Guardar métricas para este género
            tasas_por_genero[genero]['accuracy'].append(accuracy)
            tasas_por_genero[genero]['fpr'].append(fpr)
            tasas_por_genero[genero]['tpr'].append(tpr)

    # Promediar las métricas por género
    for genero in df_original['Genero'].unique():
        tasas_por_genero[genero]['accuracy_promedio'] = float(sum(tasas_por_genero[genero]['accuracy']) / num_iteraciones)
        tasas_por_genero[genero]['fpr_promedio'] = float(sum(tasas_por_genero[genero]['fpr']) / num_iteraciones)
        tasas_por_genero[genero]['tpr_promedio'] = float(sum(tasas_por_genero[genero]['tpr']) / num_iteraciones)
        tasas_por_genero[genero]['tasa_seleccion_promedio'] = float(sum(tasas_por_genero[genero]['tasas_seleccion']) / num_iteraciones)

    return modelo, tasas_por_genero