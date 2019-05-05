from keras.preprocessing import image

def LoadNReshape(file, size):
    # Load the image from file
    img = image.load_img(file)

    # Turn the image into an array
    img = image.img_to_array(img)

    # Normalize the image by /255 using float32
    img =  img.astype('float32')/255

    # Reshape the image so its Keras ready
    img = img.reshape(1,size)

    return img