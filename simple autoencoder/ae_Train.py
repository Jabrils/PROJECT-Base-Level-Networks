from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

(X_train, _), (X_test, _) = mnist.load_data()
print(f"First shape:\nTraining: {X_train.shape}\nTesting: {X_test.shape}\n")

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
print(f"Before we reshape:\nTraining: {X_train.shape}\nTesting: {X_test.shape}\n")

X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))
print(f"After we reshape:\nTraining: {X_train.shape}\nTesting: {X_test.shape}\n")

input_img= Input(shape=(784,))

# Encoder Weights
encoded = Dense(units=128, activation='relu')(input_img)
encoded = Dense(units=64, activation='relu')(encoded)

# Decoder Weights
decoded = Dense(units=128, activation='relu')(encoded)
decoded = Dense(units=784, activation='sigmoid')(decoded)

# Create the Autoencoder Model
autoencoder=Model(input_img, decoded)

# If you want to load your model to continue training
#autoencoder.load_weights("model.h5")

# Can confirm to see if we have the right shape model
# autoencoder.summary()

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the autoencoder, x is our training data, & y is our training labels, which is still X because we want our autoencoder to look as close to the original input
autoencoder.fit(x=X_train,
                y=X_train,
                epochs=100,
                batch_size=512,
                shuffle=True,
                validation_data=(X_test, X_test))

autoencoder.save("model.h5")