# 🤖 Taller de Fairness en Inteligencia Artificial ⚖️👩‍💻

## 🎯 Objetivo del Taller
- 📚 Analizar implicaciones éticas y de equidad

- 🔄 Desarrollar soluciones creativas

- 🔍 Identificar oportunidades y desafíos en IA

## 🕵️ Caso de Estudio: Contratación Sesgada en un Departamento de Informática

### 🔍 Contexto
Un sistema de IA para contratación en el sector informático muestra claros sesgos de género en la selección de candidatos, reflejando y amplificando los prejuicios históricos en la industria tecnológica.

## 🧩 Actividad Principal: Análisis del Caso en Grupos

### 📋 Instrucciones de Trabajo
Los participantes se dividirán en grupos para abordar tres aspectos fundamentales del sesgo en IA:

1. **🕵️‍♀️ Identificación de Fuentes de Sesgo**
   - Examinar los datos de entrenamiento
   - Detectar patrones de discriminación
   - Analizar la representatividad de los datos

2. **🔬 Análisis de Amplificación de Sesgos**
   - Comprender cómo el modelo reproduce y potencia los sesgos existentes
   - Identificar mecanismos de perpetuación de desigualdades
   - Evaluar el impacto en los procesos de selección

3. **🛠️ Propuestas de Mitigación**
   - Desarrollar estrategias para reducir sesgos
   - Proponer mejoras en la recopilación y tratamiento de datos
   - Diseñar métodos de corrección algorítmica

## 📅 Agenda del Taller

### 1. 🧠 Mapa Mental de Desafíos Éticos (Fase de Clarificación) - 10 minutos
- Unirse a grupos de trabajo
- Identificar desafíos éticos en IA:
  * 🔍 Transparencia
  * 🌈 Diversidad
  * ⚖️ Responsabilidad
  * 🔒 Privacidad

### 2. 💡 Brainwriting 6-3-5 (Fase de Ideación) - 5 minutos
- **6 personas** > **3 ideas** > en **5 minutos**

>  ¿Cómo abordar desafíos éticos en la construcción y ajuste de modelos de machine learning  - inteligencia artificial?


### 3. 🚀 Prototipado Rápido (Fase de Desarrollo) - 10 minutos
- Desarrollar un prototipo de IA ética
- Consideraciones clave:
  * 📊 Diversidad de datos
  * 🔍 Transparencia
  * 🌍 Impacto social

### 4. 🗣️ Presentación de Prototipos (Fase de Implementación) - 10 minutos
- Cada grupo presenta su prototipo en **1-2 minutos**

## 🧠 Consideraciones Finales
- Reflexionar sobre la responsabilidad ética en el desarrollo de IA
- Comprender la importancia de la equidad y la inclusión
- Desarrollar estrategias prácticas para mitigar sesgos

*Nota: Este taller busca fomentar una reflexión crítica sobre los sesgos en sistemas de inteligencia artificial, promoviendo un desarrollo tecnológico más justo e inclusivo.* 🌟🤝

# 📚 Material complementario: Simulación de Análisis de Factores para la Contratación de Personal

Este repositorio contiene un proyecto de análisis de datos con el objetivo de explorar los factores que influyen en la **contratación** de personal dentro de una empresa de tecnología. 🖥️ Utilizando técnicas de análisis estadístico y de machine learning, este proyecto busca identificar patrones y correlaciones entre diversas características de los candidatos (edad, género, años de experiencia, habilidades, salario previo, entre otros) y la variable de **Contratación**. 🔍

## 🎯 Objetivos del Caso
- **Explorar** y **preprocesar** datos relacionados con la contratación de personal. 🕵️‍♀️
- **Analizar** la relación entre diversas variables (como género, edad, años de experiencia) y la variable objetivo **Contratacion**. 📊
- **Construir modelos de machine learning** para predecir la contratación de un candidato, evaluando el impacto de las variables sensibles (como género) y aplicando estrategias de fairness (justicia algorítmica). 🤖
- **Visualizar** las distribuciones y correlaciones de las variables mediante herramientas como *pairplots*, matrices de correlación, etc. 📈

## 🔬 Descripción del Proyecto
### 1. 🔍 **Análisis Exploratorio de Datos (EDA)**
   - Limpieza y transformación de los datos. 🧹
   - Análisis descriptivo de las variables, incluyendo estadísticas clave como medias, desviaciones estándar, y correlaciones entre las características. 📉
   - Visualización de las distribuciones y relaciones entre las variables usando gráficos de dispersión, histogramas y mapas de calor. 🌈

### 2. 🛠️ **Preprocesamiento de Datos**
   - Conversión de variables categóricas a variables numéricas utilizando técnicas como `one-hot encoding`. 🔢
   - Tratamiento y normalización de las atributos. ✨

### 3. 🤖 **Modelos de Machine Learning**
   - Creación de modelos predictivos como **Logistic Regression**, evaluando el rendimiento de cada modelo mediante métricas de error. 🧮
   - Implementación de penalizaciones de **fairness** (justicia algorítmica) en los modelos para asegurarse de que no haya sesgos hacia ciertos grupos (por ejemplo, género). ⚖️
   - Comparación de modelos que excluyen o ajustan variables sensibles para evaluar el impacto en la equidad y precisión de las predicciones. 🤝

### 4. 🔢 **Métricas de Fairness**
   - Cálculo de disparidades en precisión, tasa de falsos positivos y tasa de falsos negativos entre diferentes grupos (por ejemplo, hombres, mujeres y personas no binarias). 📊
   - Evaluación de cómo las diferentes estrategias de penalización afectan el rendimiento y la equidad de los modelos. 🌟

## 📌 Requisitos
- Python 3.7 o superior 🐍
- Librerías necesarias:
  - `pandas` 📊
  - `numpy` 📐
  - `scikit-learn` 🤖
  - `seaborn` 🌈
  - `matplotlib` 📈
  - `scipy` 🧮

## 🚀 Instalación

1. Clona este repositorio:

```bash
   git clone https://github.com/tu_usuario/analisis-contratacion.git
```

# 🌐 Métricas Clave en Fairness

Las métricas FPR y TPR son fundamentales para la evaluación de fairness en modelos de aprendizaje automático porque permiten identificar y corregir posibles sesgos en el rendimiento del modelo hacia diferentes grupos. Estas métricas ayudan a asegurarse de que el modelo no esté favoreciendo ni perjudicando injustamente a ningún grupo, lo cual es esencial para el uso ético y justo de la inteligencia artificial.

## Fairness en ML
El objetivo es desarrollar modelos de aprendizaje automático que sean justos y no discriminen a ciertos grupos de personas. Esto es particularmente importante cuando los modelos se usan en áreas como contratación, justicia penal, salud, etc., donde las decisiones basadas en estos modelos pueden tener grandes implicaciones.

## FPR y TPR en Fairness:
- TPR (True Positive Rate), también conocido como sensibilidad o recall, mide la capacidad del modelo para identificar correctamente a los positivos reales de cada grupo.
- FPR (False Positive Rate) mide la proporción de elementos negativos que fueron incorrectamente clasificados como positivos. En términos de equidad, una alta tasa de falsos positivos podría ser problemática si un grupo está siendo penalizado erróneamente.
  
## Fairness y Equidad en el Rendimiento
Las métricas de FPR y TPR se utilizan en fairness porque permiten medir disparidades en el rendimiento del modelo entre diferentes subgrupos. En otras palabras, estas métricas se enfocan en si el modelo está tratando de manera equitativa a los diferentes grupos. Ejemplos de métricas de fairness basadas en FPR y TPR

1. Demographic Parity:
   - Busca que un modelo tome decisiones de manera equitativa entre grupos, sin que algún grupo tenga más probabilidades de ser seleccionado que otro.
   - Para esto, podríamos comparar las tasas de TPR o FPR entre grupos. Si un grupo tiene una tasa significativamente más alta de TPR que otro, se podría concluir que el modelo está favoreciendo a ese grupo en términos de identificación de positivos.
2. Equal Opportunity:
   - Una variante más fuerte de Demographic Parity se basa en TPR: los modelos deben ser igualmente buenos para identificar positivos entre los diferentes grupos. Si un grupo tiene una TPR significativamente más baja, el modelo no está siendo justo.
   - En este caso, lo ideal sería que la tasa de verdaderos positivos (TPR) para cada grupo fuera similar (es decir, el modelo debería ser igual de efectivo para detectar verdaderos positivos en todos los grupos).

3. Equalized Odds:
   - Esta es una condición de fairness que asegura que tanto TPR como FPR sean iguales entre los diferentes grupos.
   - Esto implica que el modelo debe tener tanto una tasa de verdaderos positivos como una tasa de falsos positivos iguales entre grupos. Si un grupo tiene una FPR más alta o una TPR más baja, esto indica que el modelo es sesgado en contra de ese grupo.
   - Si FPR y TPR son desbalanceados entre grupos, esto sugiere que el modelo podría estar favoreciendo o perjudicando injustamente a un grupo en particular.

4. Fairness Through Unawareness:
   - A veces se usa una versión de fairness en la que se asegura que el modelo no utiliza características sensibles (como el género o la raza) directamente. Sin embargo, aún pueden surgir disparidades en FPR y TPR entre diferentes grupos, incluso si esas características no se usan directamente.

### ¿Por qué FPR y TPR son clave en fairness?

- Equidad en las decisiones: Las métricas de FPR y TPR son esenciales porque miden el rendimiento diferencial del modelo entre diferentes grupos. Si el modelo tiene un alto FPR para un grupo pero no para otro, podría indicar que el modelo está discriminando injustamente contra ese grupo, generando más falsos positivos para ese grupo.
- Impacto en grupos desproporcionados: Por ejemplo, en un sistema de justicia penal automatizado, un modelo que tenga una alta FPR (falsos positivos) para una minoría podría generar más encarcelamientos erróneos para ese grupo, lo que sería un perjuicio para su bienestar. Del mismo modo, si un grupo tiene una baja TPR (es decir, el modelo no identifica correctamente a los individuos positivos de ese grupo), esto podría significar que están siendo injustamente desatendidos por el modelo.

### Aplicación de FPR y TPR en fairness

Al evaluar un modelo para fairness, los FPR y TPR pueden utilizarse de la siguiente manera:
-  Si un modelo tiene una TPR significativamente más baja para un grupo (por ejemplo, mujeres o personas no binarias), esto indica que el modelo no está reconociendo adecuadamente a los miembros de ese grupo cuando deberían ser clasificados como positivos (por ejemplo, identificando correctamente a personas que califican para un beneficio o programa).
-  Si un modelo tiene una FPR significativamente más alta para un grupo, significa que hay más casos de falsos positivos para ese grupo, lo que podría resultar en penalizaciones incorrectas o tratamientos adversos.


<br>

---- 
*¡Un proyecto dedicado al uso y generación de modelos de AI-ML más justos y equitativos 🌍🤝*
---
