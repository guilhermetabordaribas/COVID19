import numpy as np
import os
import random

from PIL import Image

DATADIR = '/storage/cpperico/HackCovid'
CATEGORIES = ['covid','normal','outros']

training_data = []
IMG_SIZE = 300
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = np.array(Image.open(os.path.join(path,img)).convert('L').resize((IMG_SIZE,IMG_SIZE)))
                training_data.append([img_array, class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle

pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()
