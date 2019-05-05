from keras.preprocessing import image
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model, load_model
import numpy as np
import matplotlib.pyplot as plt

(X_train, _), (X_test, _) = mnist.load_data()
X_test = X_test.astype('float32')/255

X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))

autoencoder = load_model("model.h5")

# Adding a test digit
img = image.load_img("myTestDigit.png", color_mode="grayscale")
img = image.img_to_array(img)
img = img.astype('float32')/255
img = img.reshape(1,784)
X_test[0] = img

# Adding a test letter
img = image.load_img("myTestLetter.png", color_mode="grayscale")
img = image.img_to_array(img)
img = img.astype('float32')/255
img = img.reshape(1,784)
X_test[1] = img

predicted = autoencoder.predict(X_test)

# Test & Predict
num_images=32
#np.random.seed(13)
random_test_images=np.random.randint(X_test.shape[0], size=num_images)

plt.figure(figsize=(18, 4))
for i, image_idx in enumerate(random_test_images):
    # display original
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(predicted[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()