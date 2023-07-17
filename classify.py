import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions

model = MobileNetV2(weights='imagenet')

#'/home/timi/images-/fish2.jpeg'
#'/home/timi/Desktop/grey1.png'
#'/home/timi/Desktop/fishcnn.jpg'
#'/home/timi/Desktop/wavyfish2.jpeg'

image_path = '/home/timi/Desktop/wavyfish.jpeg'
image = tf.keras.utils.load_img(image_path)

image = image.resize((224,224))

data = np.empty((1, 224, 224, 3))
data[0] = image

data = preprocess_input(data)

predictions = model.predict(data)
print('Shape: {}'.format(predictions.shape))

output_neuron = np.argmax(predictions[0])
print('Most active neuron: {} ({:.2f}%)'.format(
    output_neuron,
    100 * predictions[0][output_neuron]
))

for name, desc, score in decode_predictions(predictions)[0]:
    print('- {} ({:.2f}%%)'.format(desc, 100 * score)) 
    