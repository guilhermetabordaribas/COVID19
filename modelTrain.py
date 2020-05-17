import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

import numpy as np

import pickle

pickle_in = open('/storage/guilherme/xrays/X.pickle', 'rb')
X = pickle.load(pickle_in)

pickle_in = open('/storage/guilherme/xrays/y.pickle', 'rb')
y = pickle.load(pickle_in)

X = X/255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', 
             optimizer='adam',
             metrics=['accuracy'])

# O np.array(y_..) tive que colocar pra rodar na fenix
history = model.fit(X_train, np.array(y_train), epochs=30, batch_size=15, 
                    validation_data=(X_test, np.array(y_test) ))

test_loss, test_acc = model.evaluate(X,  np.array(y), verbose=2)


report = []
report.append([history.history['accuracy'], history.history['val_accuracy']])
report.append([test_acc, test_loss])

pickle_out = open('/storage/guilherme/xrays/report.pickle', 'wb')
pickle.dump(report, pickle_out)
pickle_out.close()

print(test_loss, test_acc)
print(history)
