
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import multiply
from tensorflow.keras.layers import Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.nn import relu6


class MobileNet_V3(object):
	def __init__(self, size, input_shape, n_outputs=2):
		self.input_shape = input_shape
		self.n_outputs = n_outputs
		self.size = size

		get_custom_objects().update({"custom_activation": Activation(self.hswish)})

	def hswish(self, layer):
		return layer * relu6(layer + 3) / 6

	def convolution_normalization_activation(self, layer, filter, kernel, stride, padding, activation):
		layer = Convolution2D(filters=filter, kernel_size=kernel, strides=stride, padding=padding)(layer)
		layer = BatchNormalization()(layer)
		
		if activation == "relu":
			layer = Activation("relu")(layer)
		elif activation == "hswish":
			layer = Activation(self.hswish)(layer)

		return layer

	def squeeze_excitation(self, inputs, filter, ratio=2):
		layer = GlobalAveragePooling2D()(inputs)

		layer = Dense(units=filter // ratio)(layer)
		layer = Activation("relu")(layer)

		layer = Dense(units=filter)(layer)
		layer = Activation("hard_sigmoid")(layer)

		layer = multiply([inputs, layer])

		return layer

	def depthwise_convolution_normalization_activation(self, layer, kernel, stride, padding, activation):
		layer = DepthwiseConv2D(kernel_size=kernel, strides=stride, padding=padding, depth_multiplier=1)(layer)
		layer = BatchNormalization()(layer)
		
		if activation == "relu":
			layer = Activation("relu")(layer)
		elif activation == "hswish":
			layer = Activation(self.hswish)(layer)

		return layer

	def bottleneck(self, inputs, outputs, kernel, stride, filter, padding, activation, use_se, shortcut):
		layer = self.convolution_normalization_activation(inputs, filter, (1, 1), (1, 1), padding, activation)

		layer = self.depthwise_convolution_normalization_activation(layer, kernel, stride, padding, activation)

		if use_se == True:
			layer = self.squeeze_excitation(layer, filter)

		layer = Convolution2D(filters=outputs, kernel_size=(1, 1), strides=(1, 1), padding="same")(layer)

		layer = BatchNormalization()(layer)

		if stride == (1, 1) and layer.get_shape().as_list()[3] == inputs.get_shape().as_list()[3]:
			layer = Add()([layer, inputs])
		
		return layer

	def mobilenet_v3_large(self):
		inputs = Input(self.input_shape)

		net = self.convolution_normalization_activation(inputs, 16, (3, 3), (2, 2), "same", "hswish")

		net = self.bottleneck(net, 16, (3, 3), (1, 1), 16, "same", "relu", False, False)
		net = self.bottleneck(net, 24, (3, 3), (2, 2), 64, "same", "relu", False, False)
		net = self.bottleneck(net, 24, (3, 3), (1, 1), 72, "same", "relu", False, True)
		net = self.bottleneck(net, 40, (5, 5), (2, 2), 72, "same", "relu", True, False)
		net = self.bottleneck(net, 40, (5, 5), (1, 1), 120, "same", "relu", True, True)
		net = self.bottleneck(net, 40, (5, 5), (1, 1), 120, "same", "relu", True, True)
		net = self.bottleneck(net, 80, (3, 3), (2, 2), 240, "same", "relu", False, False)
		net = self.bottleneck(net, 80, (3, 3), (1, 1), 200, "same", "hswish", False, True)
		net = self.bottleneck(net, 80, (3, 3), (1, 1), 184, "same", "hswish", False, True)
		net = self.bottleneck(net, 80, (3, 3), (1, 1), 184, "same", "hswish", False, True)
		net = self.bottleneck(net, 112, (3, 3), (1, 1), 480, "same", "hswish", True, False)
		net = self.bottleneck(net, 112, (3, 3), (1, 1), 672, "same", "hswish", True, True)
		net = self.bottleneck(net, 160, (5, 5), (2, 2), 672, "same", "hswish", True, False)
		net = self.bottleneck(net, 160, (5, 5), (1, 1), 672, "same", "hswish", True, True)
		net = self.bottleneck(net, 160, (5, 5), (1, 1), 960, "same", "hswish", True, True)

		net = self.convolution_normalization_activation(net, 960, (3, 3), (1, 1), "same", "hswish")

		net = AveragePooling2D(pool_size=(7, 7), padding="valid")(net)

		net = Convolution2D(1280, (3, 3), (1, 1), padding="same", activation="relu")(net)
		
		net = Convolution2D(self.n_outputs, (1, 1), (1, 1), padding="valid")(net)
		net = self.hswish(net)

		net = Flatten()(net)
		net = Softmax()(net)

		model = Model(inputs=inputs, outputs=net)

		print("MOBILENETV3 ARCHITECTURE")

		model.summary()

		return model
