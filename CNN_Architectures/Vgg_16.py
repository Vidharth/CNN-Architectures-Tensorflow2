
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential


class VGG16(object):
	def __init__(self, input_shape=(224, 224, 3), n_outputs=2):
		self.model = Sequential()
		self.input_shape = input_shape
		self.n_outputs = n_outputs

	def vgg_16_architecture(self):
		self.model.add(InputLayer(input_shape=self.input_shape))
		
		self.model.add(Convolution2D(filters=64,kernel_size=(3,3), padding="same"))
		self.model.add(Activation("relu"))

		self.model.add(Convolution2D(filters=64,kernel_size=(3,3), padding="same"))
		self.model.add(Activation("relu"))

		self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

		self.model.add(Convolution2D(filters=128, kernel_size=(3,3), padding="same"))
		self.model.add(Activation("relu"))

		self.model.add(Convolution2D(filters=128, kernel_size=(3,3), padding="same"))
		self.model.add(Activation("relu"))

		self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

		self.model.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same"))
		self.model.add(Activation("relu"))
		self.model.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same"))
		self.model.add(Activation("relu"))
		self.model.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same"))
		self.model.add(Activation("relu"))

		self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

		self.model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same"))
		self.model.add(Activation("relu"))
		self.model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same"))
		self.model.add(Activation("relu"))
		self.model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same"))
		self.model.add(Activation("relu"))

		self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

		self.model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same"))
		self.model.add(Activation("relu"))
		self.model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same"))
		self.model.add(Activation("relu"))
		self.model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same"))
		self.model.add(Activation("relu"))

		self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

		self.model.add(Flatten())
		
		self.model.add(Dense(units=4096))
		self.model.add(Activation("relu"))

		self.model.add(Dense(units=4096))
		self.model.add(Activation("relu"))

		self.model.add(Dense(units=self.n_outputs, activation="softmax"))

		return self.model
	
	def build_model(self):
		self.model = self.vgg_16_architecture()

		print("VGG16 ARCHITECTURE")

		self.model.summary()

		return self.model
