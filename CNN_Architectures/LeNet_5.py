
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import Sequential


class LeNet5(object):
	def __init__(self, input_shape=(32, 32, 3), n_outputs=2):
		self.model = Sequential()
		self.input_shape = input_shape
		self.n_outputs = n_outputs

	def lenet_5_architecture(self):
		self.model.add(InputLayer(input_shape=self.input_shape))
		
		self.model.add(Convolution2D(filters=6, kernel_size=(3, 3)))
		self.model.add(Activation("tanh"))

		self.model.add(AveragePooling2D())

		self.model.add(Convolution2D(filters=16, kernel_size=(3, 3)))
		self.model.add(Activation("tanh"))

		self.model.add(AveragePooling2D())

		self.model.add(Flatten())

		self.model.add(Dense(units=128, activation='tanh'))
		self.model.add(Activation("tanh"))

		self.model.add(Dense(units=self.n_outputs, activation = 'softmax'))

		return self.model
	
	def build_model(self):
		self.model = self.lenet_5_architecture()

		print("LENET5 ARCHITECTURE")

		self.model.summary()

		return self.model
