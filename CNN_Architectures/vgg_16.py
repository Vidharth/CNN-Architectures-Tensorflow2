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
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model


class VGG16(Model):
	def __init__(self, classes):
		super(VGG16, self).__init__()

		self.conv_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv_1")
		self.conv_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv_2")
		self.max_pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_1")
		self.conv_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv_3")
		self.conv_4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv_4")
		self.max_pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_2")
		self.conv_5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv_5")
		self.conv_6 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv_6")
		self.conv_7 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv_7")
		self.max_pool_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_3")
		self.conv_8 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv_8")
		self.conv_9 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv_9")
		self.conv_10 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv_10")
		self.max_pool_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_4")
		self.conv_11 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv_11")
		self.conv_12 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv_12")
		self.conv_13 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv_13")
		self.max_pool_5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_5")
		self.flatten = Flatten()
		self.fc_1 = Dense(units=4096, activation="relu", name="fc_1")
		self.fc_2 = Dense(units=4096, activation="relu", name="fc_2")
		self.fc_3 = Dense(units=classes, activation="softmax", name="output_fc_3")

	def call(self, inputs):
		x = self.conv_1(inputs)
		x = self.conv_2(x)
		x = self.max_pool_1(x)
		x = self.conv_3(x)
		x = self.conv_4(x)
		x = self.max_pool_2(x)
		x = self.conv_5(x)
		x = self.conv_6(x)
		x = self.conv_7(x)
		x = self.max_pool_3(x)
		x = self.conv_8(x)
		x = self.conv_9(x)
		x = self.conv_10(x)
		x = self.max_pool_4(x)
		x = self.conv_11(x)
		x = self.conv_12(x)
		x = self.conv_13(x)
		x = self.max_pool_5(x)
		x = self.flatten(x)
		x = self.fc_1(x)
		x = self.fc_2(x)
		x = self.fc_3(x)

		return x
