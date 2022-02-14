__author__ = "Hari Vidharth"
__license__ = "Open Source"
__version__ = "1.0"
__maintainer__ = "Hari Vidharth"
__email__ = "viju1145@gmail.com"
__date__ = "Feb 2020"
__status__ = "Prototype"


import os
import gc

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from tensorflow.keras import backend
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input
from vgg_16 import VGG16


gc.collect()
backend.clear_session()

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

((X_train, Y_train), (X_test, Y_test)) = cifar10.load_data()

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

lb = LabelBinarizer()
Y_train = lb.fit_transform(Y_train)
Y_test = lb.transform(Y_test)

network = VGG16(10)
network.build(input_shape = (None, 32, 32, 3))
network.call(Input(shape = (32, 32, 3)))
network.summary()

network.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

network.fit(x=X_train, y=Y_train, batch_size=64, epochs=2, validation_split=0.2, shuffle=True, use_multiprocessing=True)

predictions = network.predict(X_test)

print(classification_report(Y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

gc.collect()
backend.clear_session()
