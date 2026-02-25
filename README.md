# 🤖 Taller: Ética y Fairness en Inteligencia Artificial

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Taller presentado en ChileWiC 2024 - XII Encuentro de Mujeres en Computación**

## 📋 Descripción

Este repositorio contiene los materiales del taller **"IA, Ética y Fairness: Un Caso de Sesgo de Selección"** presentado en [ChileWiC 2024](https://chilewic.cl/2024/11/23/ya-esta-aqui-conoce-el-programa-completo-de-chilewic-2024/) (XII Encuentro de Mujeres en Computación), celebrado el 29 de noviembre de 2024 en la Universidad Mayor, Santiago, Chile.

El taller aborda los desafíos éticos en aplicaciones de inteligencia artificial y machine learning, con un enfoque práctico en la identificación y mitigación de sesgos algorítmicos. Incluye un caso de estudio sobre sesgos de género en sistemas de contratación, código Python para análisis de fairness, y metodologías para evaluar y corregir disparidades en modelos de machine learning.

### 🎯 Objetivos del Taller

- Comprender los desafíos éticos en aplicaciones reales de IA
- Analizar un caso de estudio sobre sesgos en algoritmos de selección
- Implementar técnicas en Python para identificar y corregir sesgos
- Discutir soluciones prácticas para crear modelos más justos y equitativos

### 💻 Metodología

- Sesión interactiva con un caso de estudio práctico
- Trabajo en equipos pequeños utilizando Creative Problem Solving
- Discusiones grupales y presentaciones
- Análisis de código Python con ejemplos reales

### 👥 Audiencia

- Estudiantes de ciencias de la computación e ingeniería
- Profesionales en tecnología y desarrollo
- Personas interesadas en ética y fairness en IA
- **Requisitos**: Conocimiento básico de Python recomendado

## 📚 Contenido del Repositorio

```
.
├── README.md                          # Este archivo
├── CITATION.cff                       # Información de citación
├── LICENSE                            # Licencia MIT
├── requirements.txt                   # Dependencias de Python
├── Bienvenida.md                      # Guía del taller y metodología
├── Taller de Ética y Fairness en IA-ML.pdf  # Presentación del taller
└── codigos_taller/                    # Código fuente y datos
    ├── caso_comparacion_modelos.py    # Comparación de modelos con métricas de fairness
    ├── chilewic_simulacion_datos.py   # Generación de datos simulados
    ├── python_worcloud.py             # Visualización de wordcloud
    ├── WIC_Generador_de_Datos.ipynb   # Notebook Jupyter para generación de datos
    └── simulacion_contratacion_tech_chile.csv  # Dataset simulado
```

## 🚀 Instalación

### Requisitos

- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/cinthyaleonor/chilewic_taller_fairness.git
cd chilewic_taller_fairness
```

2. Crear un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 💡 Uso

### Caso de Estudio: Sesgo de Selección en Contratación

El taller incluye un caso práctico sobre sesgos de género en sistemas de contratación en el sector tecnológico. El código permite:

1. **Generar datos simulados** con sesgos intencionales:
```bash
python codigos_taller/chilewic_simulacion_datos.py
```

2. **Comparar modelos** con diferentes estrategias de mitigación de sesgos:
```bash
python codigos_taller/caso_comparacion_modelos.py
```

3. **Explorar el notebook** para análisis interactivo:
```bash
jupyter notebook codigos_taller/WIC_Generador_de_Datos.ipynb
```

### Métricas de Fairness Implementadas

- **FPR (False Positive Rate)**: Tasa de falsos positivos por grupo
- **TPR (True Positive Rate)**: Tasa de verdaderos positivos (sensibilidad)
- **Disparidad en FPR**: Diferencia en tasas de falsos positivos entre grupos
- **Demographic Parity**: Paridad demográfica en las decisiones
- **Equal Opportunity**: Igualdad de oportunidades entre grupos

## 📖 Citación

Si utiliza este material en su investigación o trabajo, por favor cite como:

```bibtex
@software{vergara2024taller,
  title = {Taller de Ética y Fairness en Inteligencia Artificial: Un Caso de Sesgo de Selección},
  author = {Vergara, Cinthya},
  year = {2024},
  month = {11},
  version = {1.0.0},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/cinthyaleonor/chilewic_taller_fairness}
}
```

O use el archivo `CITATION.cff` incluido en el repositorio.

## 🔗 Referencias y Enlaces

- [Programa completo de ChileWiC 2024](https://chilewic.cl/2024/11/23/ya-esta-aqui-conoce-el-programa-completo-de-chilewic-2024/)
- [Noticia sobre ChileWiC 2024 - IMFD](https://imfd.cl/cerca-de-200-participantes-en-chilewic-2024/)
- [Asociación Chilena de Mujeres en Inteligencia Artificial](https://www.mujeresia.cl/)
- [ChileWiC - Women in Computing Chile](https://chilewic.cl/)

## 👤 Autora

**Cinthya Vergara**  
Experta en Analítica de Datos  
Email: civergara@alumnos.uai.cl  
GitHub: [@cinthyaleonor](https://github.com/cinthyaleonor)  
Zenodo: [@civergara](https://zenodo.org/search?q=owner%3Acivergara)

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

Este taller fue presentado en el marco de **ChileWiC 2024**, organizado por académicas de diversas universidades de Chile, con el apoyo de NIC Chile, ISCI, Cenia e IMFD.

---

**Evento**: ChileWiC 2024 - XII Encuentro de Mujeres en Computación  
**Fecha**: 29 de noviembre de 2024  
**Lugar**: Universidad Mayor, Auditorio, Manuel Montt 367, Providencia, Santiago, Chile  
**Duración**: 60 minutos

#IAÉtica #Fairness #TallerPráctico #InteligenciaArtificial #chilewic #ACM #stem #ResponsibleAI

