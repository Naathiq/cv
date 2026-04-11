import tensorflow as tf

# 1. Load & Normalize Data (MNIST)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Build the Deep Network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), 
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dropout(0.2),                  
    tf.keras.layers.Dense(10, activation='softmax')
])

# 3. Compile and Train
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 4. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print('Accuracy:', acc)