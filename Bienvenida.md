# 🤖✨ Taller de Fairness en Inteligencia Artificial

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

*¡Un proyecto dedicado a hacer la contratación más justa y equitativa! 🌍🤝*
