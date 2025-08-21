# Detección de Neumonía con Deep Learning usando Radiografías de Tórax

![Banner de Radiografía](https://images.unsplash.com/photo-1579656233139-3f26a1b538f2?q=80&w=2070&auto=format&fit=crop)
*Una herramienta de IA para asistir en el diagnóstico clínico.*

## Resumen del Proyecto 🏥💻

Este proyecto utiliza una Red Neuronal Convolucional (CNN) a través de *Transfer Learning* para clasificar radiografías de tórax, distinguiendo entre pacientes sanos y aquellos con neumonía. Como **enfermero e ingeniero de software**, mi objetivo fue construir una herramienta que no solo sea técnicamente sólida, sino también clínicamente relevante, segura y fácil de interpretar para profesionales de la salud.

---

## El Problema Clínico

El diagnóstico de la neumonía a través de radiografías de tórax es una tarea común pero crítica, donde la rapidez y la precisión pueden impactar significativamente el tratamiento del paciente. Un sistema automatizado de ayuda diagnóstica puede servir como una valiosa segunda opinión, especialmente en entornos de alta demanda, para ayudar a priorizar casos y reducir la carga de trabajo del personal clínico.

---

## Dataset

El proyecto utiliza el dataset **"Chest X-Ray Images (Pneumonia)"** disponible en [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Contiene 5,863 imágenes de radiografías de tórax pediátricas pre-etiquetadas y divididas en conjuntos de entrenamiento, validación y prueba.

---

## Metodología y Solución Técnica

El pipeline del proyecto se desarrolló en Python utilizando un script automatizado (`train_model.py`) y se compone de los siguientes pasos:

1.  **Preprocesamiento:** Las imágenes fueron redimensionadas a 224x224 píxeles y sus valores normalizados al rango [0, 1]. Se aplicó **Aumento de Datos** (*Data Augmentation*) al conjunto de entrenamiento (rotaciones, zooms, giros) para mejorar la generalización del modelo y reducir el sobreajuste.

2.  **Modelo:** Se implementó una estrategia de **Transfer Learning** utilizando la arquitectura **VGG16** (pre-entrenada en ImageNet). Las capas convolucionales base se "congelaron" para conservar su conocimiento, y se añadieron nuevas capas densas personalizadas para la clasificación binaria (Normal vs. Neumonía).

3.  **Entrenamiento:** El modelo fue entrenado en un entorno con GPU. Se compiló con el optimizador Adam (`learning_rate=0.0001`) y la función de pérdida `binary_crossentropy`, ideal para problemas de clasificación binaria.

---

## Resultados y Análisis Clínico

El modelo final alcanzó una **precisión general del 91%** en el conjunto de prueba, que nunca antes había visto. Sin embargo, el análisis más importante es el de su comportamiento como herramienta diagnóstica.

![Resultados del Modelo](matriz_de_confusion.png)

### Interpretación Clínica (Mi Aporte como Enfermero):
* **Alta Sensibilidad (Recall) para Neumonía (98%):** Este es el resultado más importante. El modelo es extremadamente eficaz para identificar a los pacientes que **realmente tienen la enfermedad**, minimizando los peligrosos **Falsos Negativos**. En la práctica, esto significa que es muy poco probable que el sistema pase por alto a un paciente enfermo.

* **Comportamiento Cauteloso:** El modelo tiende a ser precavido, generando algunos **Falsos Positivos** (pacientes sanos clasificados como enfermos). Este tipo de error es clínicamente más seguro, ya que simplemente implicaría una segunda revisión por parte de un humano, en lugar de enviar a casa a un paciente que requiere tratamiento.

Este perfil de rendimiento hace que el modelo sea una **excelente y segura herramienta de cribado (*screening*) o de segunda opinión.**

---

## Cómo Usar este Proyecto

Para replicar este proyecto, sigue estos pasos:

1.  **Clona el repositorio:**
    ```bash
    git clone [https://github.com/solidsaul96/Deteccion-Neumonia-con-Deep-Learning.git](https://github.com/solidsaul96/Deteccion-Neumonia-con-Deep-Learning.git)
    cd Deteccion-Neumonia-con-Deep-Learning
    ```

2.  **Configura tu API de Kaggle:** Asegúrate de tener tu archivo `kaggle.json` en la carpeta `~/.kaggle/` para la autenticación automática.

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecuta el script de entrenamiento:**
    ```bash
    python train_model.py
    ```
    El script se encargará de todo: descargar los datos, entrenar, evaluar y guardar el modelo final (`modelo_detector_neumonia_final.h5`).

    

---

## Futuras Mejoras

* **Crear un Set de Validación más Robusto:** Dividir el set de entrenamiento para obtener una métrica de validación más estable durante el entrenamiento.
* **Experimentar con otras Arquitecturas:** Probar modelos como ResNet50 o InceptionV3.
* **Despliegue:** Empaquetar el modelo en una API con Flask/FastAPI y crear una interfaz web simple con Streamlit para permitir la subida y clasificación de imágenes en tiempo real.


---

## Modelo Pre-entrenado

Debido a su tamaño, el modelo entrenado (`.h5`) no está incluido en este repositorio. Puedes descargarlo desde el siguiente enlace:

* **[Descargar Modelo (modelo_detector_neumonia_final.h5)](https://drive.google.com/file/d/1--n_HWd-pPHyBFjh0Euloh3crBje1Iav/view?usp=sharing)**

Una vez descargado, colócalo en la carpeta principal del proyecto para poder usarlo en otros scripts.

---

---

## Contacto

* **Saul Alejandro Medina Diaz** - [LinkedIn](https://www.linkedin.com/in/saul-alejandro-medina-diaz-289444363) | [GitHub](https://github.com/solidsaul96)
