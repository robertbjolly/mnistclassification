import numpy as np
import codecs
import shutil
import gzip
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle


files = ['train-images-idx3-ubyte.gz',
       'train-labels-idx1-ubyte.gz',
       't10k-images-idx3-ubyte.gz',
       't10k-labels-idx1-ubyte.gz']

# The downloaded files are in an archive format and needs to be extracted
for file in files:
    with gzip.open(file, 'rb') as f_in:
        with open(file.split('.')[0], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


# CONVERTS 4 BYTES TO A INT
def get_int(b):   # CONVERTS 4 BYTES TO A INT
	encoded = codecs.encode(b, 'hex')
	#print(encoded)# CONVERTS 4 BYTES TO A INT
	return int(encoded, 16)


new_files = ['train-labels-idx1-ubyte',
       'train-images-idx3-ubyte',
       't10k-images-idx3-ubyte',
       't10k-labels-idx1-ubyte']


data_dict = {}

for file in new_files:
  with open(file, 'rb') as f:
      data = f.read()
      file_type = get_int(data[:4])
      
      if file_type == 2051:
        file_type = 'images'
        length = get_int(data[4:8])  # 4-7: LENGTH OF THE ARRAY  (DIMENSION 0)
        num_rows = get_int(data[8:12])
        num_columns = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16) # READ THE LABEL VALUES AS INTEGERS
        parsed = parsed.reshape(length,num_rows,num_columns)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES x HEIGHT x WIDTH]
        if length == 60000:
          type_set = 'train'
        else:
          type_set = 'test'
        data_dict[f"{type_set}_{file_type}"] = parsed
        

      elif file_type == 2049:
        file_type = 'labels'
        length = get_int(data[4:8])  # 4-7: LENGTH OF THE ARRAY  (DIMENSION 0)
        num_items = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8) # READ THE LABEL VALUES AS INTEGERS
        parsed = parsed.reshape(length)
        if length == 60000:
          type_set = 'train'
        else:
          type_set = 'test'
        data_dict[f"{type_set}_{file_type}"] = parsed


train_images = data_dict.get('train_images')
train_labels = data_dict.get('train_labels')
test_images = data_dict.get('test_images')
test_labels = data_dict.get('test_labels')

'''
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

plt.imshow(train_images[0], cmap = plt.cm.binary)
plt.show()

'''

# Scaled values (0-255) between 0 and 1, making it easier for network to learn
train_images = tf.keras.utils.normalize(train_images, axis=1) 
test_images = tf.keras.utils.normalize(test_images, axis=1) 
print(np.shape(train_images[0]))


number_mnist = (train_images, train_labels), (test_images, test_labels)


# Sequential groups a linear stack of layers into a tf.keras.Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # Input (How many units in layer, neuron fire or not fire(return 1 or 0))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # Output (Number of classifications, picks highest probablity in set)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=15)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Loss: {test_loss}, Accuracy: {test_acc}")

model.save('saved_model/my_model') 
