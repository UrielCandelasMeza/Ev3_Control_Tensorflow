import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from datetime import datetime

# Configuración inicial
input_shape = (300, 300, 3)
num_classes = 5
batch_size = 32
epochs = 50
val_split = 0.2
test_split = 0.1  # Nuevo: separación para test

train_data_dir = "Datasets"
model_name = "optimized_model.keras"
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Creación de generadores de datos con separación train/val/test
# Data augmentation solo para entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    validation_split=val_split + test_split  # Total de datos para val+test
)

# Generador para validación y test (sin aumento de datos)
val_test_datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split + test_split)

# Generador de entrenamiento (80% de los datos)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=42
)

# Generador de validación (10% de los datos)
validation_generator = val_test_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=42
)

# Generador de test (10% de los datos)
test_generator = val_test_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=42,
    shuffle=False  # Importante para evaluación consistente
)

# Cálculo de class weights para datasets desbalanceados
class_weights = compute_class_weight('balanced', 
                                   classes=np.unique(train_generator.classes), 
                                   y=train_generator.classes)
class_weights = dict(enumerate(class_weights))

# Modelo con MobileNetV2 preentrenado
base_model = tf.keras.applications.MobileNetV2(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

# Congelamos las capas base inicialmente
base_model.trainable = False

# Construcción del modelo
model = Sequential([
    base_model,
    #tf.keras.layers.Resizing(224, 224),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compilación con optimizador AdamW (mejor que Adam estándar)
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=0.001,
    weight_decay=0.004  # Decoupled weight decay
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Callbacks mejorados
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        min_delta=0.001
    ),
    ModelCheckpoint(
        filepath=f'Models/{model_name}',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
]

# Entrenamiento en dos fases
# Fase 1: Entrenar solo las capas superiores
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Fase 2: Fine-tuning (descongelar algunas capas del modelo base)
base_model.trainable = True
# Descongelar solo las últimas 40 capas
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Recompilar con menor learning rate para fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continuar entrenamiento
history_fine = model.fit(
    train_generator,
    epochs=epochs + 20,  # Extender el entrenamiento
    initial_epoch=history.epoch[-1],
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Evaluación final con el conjunto de test
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
print(f"\nResultados finales en el conjunto de test:")
print(f"Precisión: {test_acc*100:.2f}%")
print(f"Pérdida: {test_loss:.4f}")
print(f"Precisión (métrica): {test_precision*100:.2f}%")
print(f"Recall: {test_recall*100:.2f}%")

# Guardar el modelo final
model.save(f'Models/final_{model_name}')