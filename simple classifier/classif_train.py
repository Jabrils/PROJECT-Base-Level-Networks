# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("comment.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:5]
Y = dataset[:,5]

# create model
model = Sequential()
model.add(Dense(8, input_dim=5, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=2500, batch_size=8)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# save the model
model.save("mod.h5")

# calculate predictions
predictions = model.predict(X)
input()