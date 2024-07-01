import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical'
    )

    return train_generator, validation_generator
