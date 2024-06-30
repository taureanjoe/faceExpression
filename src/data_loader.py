import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np

def load_data(data_dir):
    img_size = (48, 48)  # FER2013 images are 48x48
    batch_size = 32

    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical'
    )

    test_generator = datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical'
    )

    return train_generator, test_generator

# Usage
data_dir = '../data/archive'
train_gen, test_gen = load_data(data_dir)

# Print class indices to see the mapping of emotions to indices
print(train_gen.class_indices)