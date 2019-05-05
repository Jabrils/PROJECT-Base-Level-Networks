from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model, load_model
import numpy as np
import matplotlib.pyplot as plt
import random as r
import backend as B

X = np.zeros([10,28*28*3])

imgSize = 28*28*3
hmGoku = 5

X[0] = B.LoadNReshape(f"data/small/goku{r.randint(0,hmGoku-1)+1}Sm.jpg", imgSize)
X[1] = B.LoadNReshape(f"data/small/goku{r.randint(0,hmGoku-1)+1}Sm.jpg", imgSize)
X[2] = B.LoadNReshape(f"data/small/goku{r.randint(0,hmGoku-1)+1}Sm.jpg", imgSize)
X[3] = B.LoadNReshape(f"data/small/goku{r.randint(0,hmGoku-1)+1}Sm.jpg", imgSize)
X[4] = B.LoadNReshape(f"data/small/goku{r.randint(0,hmGoku-1)+1}Sm.jpg", imgSize)
X[5] = B.LoadNReshape(f"data/small/goku{r.randint(0,hmGoku-1)+1}Sm.jpg", imgSize)
X[6] = B.LoadNReshape(f"data/small/goku{r.randint(0,hmGoku-1)+1}Sm.jpg", imgSize)
X[7] = B.LoadNReshape(f"data/small/goku{r.randint(0,hmGoku-1)+1}Sm.jpg", imgSize)
X[8] = B.LoadNReshape(f"data/small/goku{r.randint(0,hmGoku-1)+1}Sm.jpg", imgSize)
X[9] = B.LoadNReshape(f"data/small/goku{r.randint(0,hmGoku-1)+1}Sm.jpg", imgSize)

autoencoder = load_model("model.h5")

predicted = autoencoder.predict(X)

# Test & Predict
num_images=10
#np.random.seed(13)
random_test_images=np.random.randint(X.shape[0], size=num_images)

plt.figure(figsize=(18, 4))
for i, image_idx in enumerate(random_test_images):
    # display original
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(X[i].reshape(28, 28, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(predicted[i].reshape(28, 28, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()