__author__ = "Hari Vidharth"
__license__ = "Open Source"
__version__ = "1.0"
__maintainer__ = "Hari Vidharth"
__email__ = "viju1145@gmail.com"
__date__ = "Feb 2020"
__status__ = "Prototype"


from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model


class VGG19():
	def __init__(self, x, y, z, o):
		self.x = x
		self. y = y
		self.z = z
		self.o = o

	def vgg19_module(self):
		self.input = Input(shape=(self.x, self.y, self.z))

		x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(self.input)
		x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)

		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)

		x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
		x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)

		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)

		x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)

		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)

		x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
		x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
		x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
		x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)

		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)

		x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
		x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
		x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
		x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)

		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)

		x = Flatten()(x)

		x = Dense(units=4096, activation="relu")(x)
		x = Dense(units=4096, activation="relu")(x)

		self.output = Dense(units=self.o, activation="softmax")(x)

	def vgg19_build(self):
		self.vgg19_module()

		network = Model(inputs=self.input, outputs=self.output)

		network.summary()

		return network
