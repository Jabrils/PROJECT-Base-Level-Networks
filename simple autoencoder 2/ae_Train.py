from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D
from keras.models import Model
import backend as B
import numpy as np

x = np.zeros([4,28*28*3])

imgSize = 28*28*3

x[0] = B.LoadNReshape("data/small/goku1Sm.jpg", imgSize)
x[1] = B.LoadNReshape("data/small/goku2Sm.jpg", imgSize)
x[2] = B.LoadNReshape("data/small/goku3Sm.jpg", imgSize)
x[3] = B.LoadNReshape("data/small/goku4Sm.jpg", imgSize)

input_img= Input(shape=(28*28*3,))

# Encoder Weights
encoded = Dense(units=28*28*3, activation='relu')(input_img)
encoded = Dense(units=16, activation='relu')(encoded)

# Decoder Weights
decoded = Dense(units=28*28*3, activation='sigmoid')(encoded)

# Create the Autoencoder Model
autoencoder=Model(input_img, decoded)

# If you want to load your model to continue training
#autoencoder.load_weights("model.h5")

# Can confirm to see if we have the right shape model
# autoencoder.summary()

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the autoencoder, x is our training data, & y is our training labels, which is still X because we want our autoencoder to look as close to the original input
autoencoder.fit(x=x,
                y=x,
                epochs=1000,
                batch_size=4,
                shuffle=True,
                validation_data=(x, x))

autoencoder.save("model.h5")