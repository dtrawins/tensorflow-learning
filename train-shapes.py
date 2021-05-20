from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    'datasets/shapes',  # This is the source directory for training images
    target_size=(28, 28),  # All images will be resized to 150x150
    batch_size=4,
    shuffle=True,
    seed=3,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary',
    color_mode='grayscale',
    save_to_dir='tmp')


print(train_generator.labels)
print(type(train_generator))

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=50,
    verbose=1)

print("model summary", model.summary())