
#| python wordcloud
#| caption: Fairness Word Cloud
#| echo: false
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# Descarga de stopwords en español
nltk.download('stopwords')

# Términos relevantes sobre fairness en IA/ML
texto_fairness = """
Fairness en IA Machine Learning equidad algoritmos sesgo discriminación transparencia ética datos entrenamiento modelo predictivo justicia representación balanceada imparcialidad clasificación análisis de impacto mitigación de sesgos diversidad inclusión 
interpretabilidad derechos humanos protección grupos vulnerables no-discriminación tratamiento contextualización compensación evaluación de impacto social responsabilidad algoritmica explicabilidad balanceo datos grupos protegidos interseccionalidad 
justicia distributiva justicia procesal Fairness IA Machine Learning equidad justicia algoritmos discriminación sesgo transparencia ética diversidad inclusión imparcialidad derechos-humanos no-discriminación igual tratamiento responsabilidad algorítmica explicabilidad datos entrenamiento datos balanceo datos representación balanceada grupos-protegidos contextualización
compensación evaluación de impacto social análisis impacto clasificación modelo predictivo interseccionalidad justicia distributiva justicia procesal protección grupos-vulnerables interpretabilidad 
"""

# Configuración del wordcloud
wordcloud = WordCloud(
    width=700, 
    height=300, 
    background_color='white', 
    stopwords=set(stopwords.words('spanish')),
    max_words=50, 
    colormap='RdPu'
).generate(texto_fairness)

# Visualización
plt.figure(figsize=(10,4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('fairness_wordcloud.png', dpi=150, bbox_inches='tight')
plt.show()

