from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    'datasets/3shapes',  # This is the source directory for training images
    target_size=(28, 28),  # All images will be resized to 150x150
    batch_size=4,
    shuffle=True,
    seed=3,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical',
    color_mode='rgb',
    save_to_dir='tmp')

test_generator = test_datagen.flow_from_directory(
    'datasets/3shapes-test',  # This is the source directory for training images
    target_size=(28, 28),  # All images will be resized to 150x150
    batch_size=4,
    shuffle=True,
    seed=3,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical',
    color_mode='rgb')

x,y = test_generator.next()

print(x)
print(y)
print(x.shape)
print(y.shape)

print(train_generator.labels)
print(type(train_generator))

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("model summary", model.summary)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=50,
    verbose=1,validation_data=test_generator)


results = model.evaluate(x,y, batch_size=1)
print("results",results)

pr = model.predict(x[0].reshape(1,28,28,3))

print("prediction",pr)
model.save("model.h5")

tf.saved_model.save(model, "saved-model/1/")



