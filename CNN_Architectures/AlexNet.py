
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Sequential


class AlexNet(object):
	def __init__(self, input_shape=(256, 256, 3), n_outputs=2):
		self.model = Sequential()
		self.input_shape = input_shape
		self.n_outputs = n_outputs

	def alex_net_architecture(self):
		self.model.add(InputLayer(input_shape=self.input_shape))
		
		self.model.add(ZeroPadding2D(padding=2))
		self.model.add(Convolution2D(filters=96, kernel_size=(11, 11), strides=(4, 4)))
		self.model.add(Activation("relu"))

		self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

		self.model.add(ZeroPadding2D(padding=2))
		self.model.add(Convolution2D(filters=256, kernel_size=(5, 5), strides=(1, 1)))
		self.model.add(Activation("relu"))

		self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

		self.model.add(ZeroPadding2D(padding=1))
		self.model.add(Convolution2D(filters=384, kernel_size=(3, 3), strides=(1, 1)))
		self.model.add(Activation("relu"))

		self.model.add(ZeroPadding2D(padding=1))
		self.model.add(Convolution2D(filters=384, kernel_size=(3, 3), strides=(1, 1)))
		self.model.add(Activation("relu"))

		self.model.add(ZeroPadding2D(padding=1))
		self.model.add(Convolution2D(filters=256, kernel_size=(3, 3), strides=(1, 1)))
		self.model.add(Activation("relu"))

		self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

		self.model.add(Flatten())

		self.model.add(Dense(units=4096))
		self.model.add(Activation("relu"))

		self.model.add(Dense(units=4096))
		self.model.add(Activation("relu"))
		
		self.model.add(Dense(units=self.n_outputs, activation="softmax"))

		return self.model
	
	def build_model(self):
		self.model = self.alex_net_architecture()

		print("ALEXNET ARCHITECTURE")

		self.model.summary()

		return self.model
