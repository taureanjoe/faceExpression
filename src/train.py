import os
from data_loader import load_data
from model import create_model
from tensorflow.keras.optimizers import Adam

# Set the path to your data directory
data_dir = '../data/archive'

# Load the data
train_gen, test_gen = load_data(data_dir)

# Create the model
model = create_model(input_shape=(48, 48, 1), num_classes=7)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    epochs=50,
    validation_data=test_gen,
    validation_steps=test_gen.samples // test_gen.batch_size
)

# Save the model
model.save('../models/emotion_model.h5')

# later add a code here to plot training history