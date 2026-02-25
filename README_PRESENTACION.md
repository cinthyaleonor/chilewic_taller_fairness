# 📊 Presentación del Taller

Este directorio contiene los materiales de la presentación del taller "IA, Ética y Fairness: Un Caso de Sesgo de Selección" presentado en ChileWiC 2024.

## Archivos

- **`Presentacion.qmd`**: Archivo fuente de Quarto (formato original)
- **`Taller de Ética y Fairness en IA-ML.pdf`**: Versión compilada en PDF

## Requisitos para Compilar

Si deseas compilar el archivo `.qmd` a HTML o PDF, necesitarás:

1. **Quarto**: Instalar desde [quarto.org](https://quarto.org/docs/get-started/)
2. **Python** con las librerías del `requirements.txt`
3. **Dependencias adicionales**:
   - `pandoc` (incluido con Quarto)
   - `revealjs` (para presentaciones HTML)

## Compilar la Presentación

### A HTML (Reveal.js):
```bash
quarto render Presentacion.qmd
```

### A PDF:
```bash
quarto render Presentacion.qmd --to pdf
```

## Notas

- La presentación utiliza código Python ejecutable (chunks de código)
- Requiere acceso a los archivos en `codigos_taller/` y el dataset CSV
- Las imágenes deben estar en la carpeta `img/` (si aplica)
- El archivo `referencias.bib` contiene las referencias bibliográficas

## Visualización Rápida

Para una visualización rápida sin compilar, consulta el archivo PDF incluido.
