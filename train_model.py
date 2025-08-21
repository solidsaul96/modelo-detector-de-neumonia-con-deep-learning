import os
import sys
import argparse
import kagglehub
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50 # Agregamos ResNet50 como opción
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================================
# --- CONFIGURACIÓN CENTRALIZADA ---
# Cualquiera puede modificar estos parámetros fácilmente en un solo lugar.
# =================================================================================
CONFIG = {
    "IMG_WIDTH": 224,
    "IMG_HEIGHT": 224,
    "BATCH_SIZE": 32,
    "EPOCHS": 10,
    "LEARNING_RATE": 0.0001,
    "BASE_MODEL": "VGG16", # Opciones: "VGG16", "ResNet50"
    "MODEL_FILENAME": "modelo_detector_neumonia_final.h5",
    "CONFUSION_MATRIX_FILENAME": "matriz_de_confusion.png"
}

# =================================================================================
# --- FUNCIONES MODULARES ---
# La lógica principal del proyecto está encapsulada en funciones.
# =================================================================================

def setup_kaggle_api():
    """Verifica que la API de Kaggle esté configurada."""
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
    if not os.path.exists(kaggle_json_path):
        print("Error: No se encontró el archivo 'kaggle.json'.")
        print(f"Por favor, asegúrate de que el archivo se encuentra en: '{kaggle_json_path}'")
        sys.exit(1)
    print("✅ API de Kaggle encontrada.")

def download_dataset():
    """Descarga el dataset y devuelve las rutas a los directorios."""
    print("Descargando y preparando el dataset...")
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    train_dir = os.path.join(path, 'chest_xray/train')
    val_dir = os.path.join(path, 'chest_xray/val')
    test_dir = os.path.join(path, 'chest_xray/test')
    print(f"Dataset listo en: {path}")
    return train_dir, val_dir, test_dir

def create_data_generators(config, train_dir, val_dir, test_dir):
    """Crea los generadores de datos para entrenamiento, validación y prueba."""
    print("Creando generadores de datos...")
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode='nearest'
    )
    validation_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(config["IMG_HEIGHT"], config["IMG_WIDTH"]),
        batch_size=config["BATCH_SIZE"], class_mode='binary', color_mode='rgb'
    )
    validation_generator = validation_test_datagen.flow_from_directory(
        val_dir, target_size=(config["IMG_HEIGHT"], config["IMG_WIDTH"]),
        batch_size=config["BATCH_SIZE"], class_mode='binary', color_mode='rgb'
    )
    test_generator = validation_test_datagen.flow_from_directory(
        test_dir, target_size=(config["IMG_HEIGHT"], config["IMG_WIDTH"]),
        batch_size=config["BATCH_SIZE"], class_mode='binary', color_mode='rgb', shuffle=False
    )
    return train_generator, validation_generator, test_generator

def build_model(config):
    """
    Construye el modelo usando Transfer Learning.
    Permite elegir entre diferentes arquitecturas base.
    """
    print(f"Construyendo el modelo con base: {config['BASE_MODEL']}...")
    input_shape = (config["IMG_HEIGHT"], config["IMG_WIDTH"], 3)
    
    # cualquier colaborador puede añadir más modelos aquí fácilmente
    if config["BASE_MODEL"] == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif config["BASE_MODEL"] == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Modelo base no soportado: {config['BASE_MODEL']}")

    for layer in base_model.layers:
        layer.trainable = False
    
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["LEARNING_RATE"]),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# =================================================================================
# --- PUNTO DE ENTRADA DEL SCRIPT ---
# Aquí se manejan los argumentos de la terminal y se ejecuta el pipeline.
# =================================================================================
if __name__ == '__main__':
    # --- Manejo de Argumentos de la Terminal ---
    # Esto permite a los usuarios sobreescribir la configuración por defecto.
    parser = argparse.ArgumentParser(description="Entrenar un modelo de CNN para detectar neumonía.")
    parser.add_argument("--epochs", type=int, help="Número de épocas para entrenar.")
    parser.add_argument("--batch_size", type=int, help="Tamaño del lote (batch size).")
    parser.add_argument("--learning_rate", type=float, help="Tasa de aprendizaje del optimizador.")
    parser.add_argument("--base_model", type=str, choices=["VGG16", "ResNet50"], help="Arquitectura base a usar.")
    
    args = parser.parse_args()

    # Actualizar CONFIG con los argumentos proporcionados por el usuario
    if args.epochs: CONFIG["EPOCHS"] = args.epochs
    if args.batch_size: CONFIG["BATCH_SIZE"] = args.batch_size
    if args.learning_rate: CONFIG["LEARNING_RATE"] = args.learning_rate
    if args.base_model: CONFIG["BASE_MODEL"] = args.base_model

    print("--- Configuración de la ejecución ---")
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
    print("------------------------------------")

    # --- Ejecución del Pipeline ---
    setup_kaggle_api()
    train_dir, val_dir, test_dir = download_dataset()
    train_gen, val_gen, test_gen = create_data_generators(CONFIG, train_dir, val_dir, test_dir)
    model = build_model(CONFIG)
    model.summary()
    
    print("\n--- Iniciando Entrenamiento ---")
    model.fit(
        train_gen,
        epochs=CONFIG["EPOCHS"],
        validation_data=val_gen,
        steps_per_epoch=train_gen.samples // CONFIG["BATCH_SIZE"],
        validation_steps=val_gen.samples // CONFIG["BATCH_SIZE"]
    )

    print("\n--- Iniciando Evaluación ---")
    loss, accuracy = model.evaluate(test_gen)
    print(f"Precisión en Prueba: {accuracy:.4f}")

    Y_pred = model.predict(test_gen)
    y_pred = (Y_pred > 0.5).astype(int).reshape(-1)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(test_gen.classes, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig(CONFIG["CONFUSION_MATRIX_FILENAME"])
    print(f"Matriz de confusión guardada como '{CONFIG['CONFUSION_MATRIX_FILENAME']}'")

    print("\n--- Reporte de Clasificación ---")
    print(classification_report(test_gen.classes, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

    model.save(CONFIG["MODEL_FILENAME"])
    print(f"\n✅ ¡Modelo final guardado como '{CONFIG['MODEL_FILENAME']}'!")