from keras.preprocessing import image
import numpy as np

x = np.zeros([2,28*28*3])

def LoadNAdd(file):
    # Load the image from file
    img = image.load_img(file)

    # Turn the image into an array
    img = image.img_to_array(img)

    # Normalize the image by /255 using float32
    img =  img.astype('float32')/255

    # Reshape the image so its Keras ready
    img = img.reshape(1,28*28*3)

    return img

x[0] = LoadNAdd("gokuSm.jpg")
x[1] = LoadNAdd("goku2Sm.jpg")

# check out its shape
print(x.shape, x)
