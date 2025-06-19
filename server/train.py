import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers

input_shape = (300, 300, 3)
num_classes = 5
batch_size = 32
epochs = 50
val_split = 0.2

train_data_dir = "Datasets"
model_name = "model2.keras"


# Datagen de entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=val_split,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# Generador de entrenamiento
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Generador de prueba
test_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

"""
model = Sequential([
  Conv2D(32,(3,3), activation='relu', input_shape=input_shape),
  MaxPooling2D(2,2),
  Dropout(0.25),

  Conv2D(64, (3,3), activation='relu'),
  MaxPooling2D(2,2),
  Dropout(0.25),

  Conv2D(128, (3,3), activation='relu'),
  MaxPooling2D(2,2),

  Flatten(),
  Dropout(0.5),

  Dense(512, activation='relu'),
  Dropout(0.5),
  Dense(num_classes, activation="softmax")
])
"""

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])


model.compile(
  optimizer=Adam(learning_rate=0.0001,weight_decay=1e-6),
  loss='categorical_crossentropy',
  metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss',patience=10, restore_best_weights=True),
    ModelCheckpoint(f'Models/{model_name}', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=5, min_lr=1e-7)
]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=test_generator,  # Usamos el test como validación
    validation_steps=test_generator.samples // batch_size,
    epochs=epochs,
    callbacks=callbacks
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\nPrecisión en el conjunto de prueba: {test_accuracy*100:.2f}%")
