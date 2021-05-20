from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import numpy as np
import datetime
#train_datagen = ImageDataGenerator(rescale=1/255)
#test_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
batch_size = 32
x = np.random.rand(batch_size,224,224,3)
x = x.astype(np.float32)

y = np.zeros((batch_size,1000))
for i in range(0,batch_size-1):
    y[i,0] = 1
y = y.astype(np.float32)

print(x)
print(y)
print(x.shape, x.dtype)
print(y.shape, y.dtype)

model = tf.keras.models.Sequential([
    #tf.keras.layers.MaxPooling2D(224, 224),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1000, activation='softmax'),
])
start_time = datetime.datetime.now()
pr = model.predict(x[0:])
end_time = datetime.datetime.now()
duration = (end_time - start_time).total_seconds() * 1000
print("predict duration", duration)
print("prediction",pr)

print("model summary", model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    x,y,
    steps_per_epoch=1,
    epochs=10,
    verbose=1)


results = model.evaluate(x,y, batch_size=1)
print("results",results)
start_time = datetime.datetime.now()
pr = model.predict(x[0:])
end_time = datetime.datetime.now()
duration = (end_time - start_time).total_seconds() * 1000
print("predict duration", duration)
print("prediction",pr)

for layer in model.layers:
    layer.trainable = False

model.save("model.h5")

tf.saved_model.save(model, "saved-model/1/")
