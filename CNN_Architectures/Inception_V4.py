
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model


class Inception_V4(object):
    def __init__(self, input_shape, n_outputs=2):
        self.input_shape = input_shape
        self.n_outputs = n_outputs

    def convolution_normalization_activation(self, layer, filter, kernel, stride, padding):
        layer = Convolution2D(filters=filter, kernel_size=kernel, strides=stride, padding=padding)(layer)
        layer = BatchNormalization()(layer)
        layer = Activation("relu")(layer)

        return layer
    
    def inception_block_A(self, input):
        branch_0 = self.convolution_normalization_activation(input, 96, (1, 1), (1, 1), "same")

        branch_1 = self.convolution_normalization_activation(input, 64, (1, 1), (1, 1), "same")
        branch_1 = self.convolution_normalization_activation(branch_1, 96, (3, 3), (1, 1), "same")

        branch_2 = self.convolution_normalization_activation(input, 64, (1, 1), (1, 1), "same")
        branch_2 = self.convolution_normalization_activation(branch_2, 96, (3, 3), (1, 1), "same")
        branch_2 = self.convolution_normalization_activation(branch_2, 96, (3, 3), (1, 1), "same")

        branch_3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(input)
        branch_3 = self.convolution_normalization_activation(branch_3, 96, (1, 1), (1, 1), "same")

        layer = concatenate([branch_0, branch_1, branch_2, branch_3])

        return layer

    def reduction_block_A(self, input):
        branch_0 = self.convolution_normalization_activation(input, 384, (3, 3), (2, 2), "valid")

        branch_1 = self.convolution_normalization_activation(input, 192, (1, 1), (1, 1), "same")
        branch_1 = self.convolution_normalization_activation(branch_1, 224, (3, 3), (1, 1), "same")
        branch_1 = self.convolution_normalization_activation(branch_1, 256, (3, 3), (2, 2), "valid")
    
        branch_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(input)

        layer = concatenate([branch_0, branch_1, branch_2])

        return layer

    def inception_block_B(self, input):
        branch_0 = self.convolution_normalization_activation(input, 384, (1, 1), (1, 1), "same")

        branch_1 = self.convolution_normalization_activation(input, 192, (1, 1), (1, 1), "same")
        branch_1 = self.convolution_normalization_activation(branch_1, 224, (1, 7), (1, 1), "same")
        branch_1 = self.convolution_normalization_activation(branch_1, 256, (1, 7), (1, 1), "same")

        branch_2 = self.convolution_normalization_activation(input, 192, (1, 1), (1, 1), "same")
        branch_2 = self.convolution_normalization_activation(branch_2, 192, (1, 7), (1, 1), "same")
        branch_2 = self.convolution_normalization_activation(branch_2, 224, (7, 1), (1, 1), "same")
        branch_2 = self.convolution_normalization_activation(branch_2, 224, (1, 7), (1, 1), "same")
        branch_2 = self.convolution_normalization_activation(branch_2, 256, (7, 1), (1, 1), "same")

        branch_3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(input)
        branch_3 = self.convolution_normalization_activation(branch_3, 128, (1, 1), (1, 1), "same")

        layer = concatenate([branch_0, branch_1, branch_2, branch_3])

        return layer

    def reduction_block_B(self, input):
        branch_0 = self.convolution_normalization_activation(input, 192, (1, 1), (1, 1), "same")
        branch_0 = self.convolution_normalization_activation(branch_0, 192, (3, 3), (2, 2), "valid")

        branch_1 = self.convolution_normalization_activation(input, 256, (1, 1), (1, 1), "same")
        branch_1 = self.convolution_normalization_activation(branch_1, 256, (1, 7), (1, 1), "same")
        branch_1 = self.convolution_normalization_activation(branch_1, 320, (7, 1), (1, 1), "same")
        branch_1 = self.convolution_normalization_activation(branch_1, 320, (3, 3), (2, 2), "valid")
    
        branch_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(input)

        layer = concatenate([branch_0, branch_1, branch_2])

        return layer
    
    def inception_block_C(self, input):
        branch_0 = self.convolution_normalization_activation(input, 256, (1, 1), (1, 1), "same")

        branch_1 = self.convolution_normalization_activation(input, 384, (1, 1), (1, 1), "same")
        branch_1_a = self.convolution_normalization_activation(branch_1, 256, (1, 3), (1, 1), "same")
        branch_1_b = self.convolution_normalization_activation(branch_1, 256, (3, 1), (1, 1), "same")
        branch_1 = concatenate([branch_1_a, branch_1_b])

        branch_2 = self.convolution_normalization_activation(input, 384, (1, 1), (1, 1), "same")
        branch_2 = self.convolution_normalization_activation(branch_2, 448, (1, 3), (1, 1), "same")
        branch_2 = self.convolution_normalization_activation(branch_2, 512, (3, 1), (1, 1), "same")
        branch_2_a = self.convolution_normalization_activation(branch_2, 256, (1, 3), (1, 1), "same")
        branch_2_b = self.convolution_normalization_activation(branch_2, 256, (3, 1), (1, 1), "same")
        branch_2 = concatenate([branch_2_a, branch_2_b])

        branch_3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(input)
        branch_3 = self.convolution_normalization_activation(branch_3, 256, (1, 1), (1, 1), "same")

        layer = concatenate([branch_0, branch_1, branch_2, branch_3])

        return layer
    
    def inception_stem(self, input):
        base = self.convolution_normalization_activation(input, 32, (3, 3), (2, 2), "valid")
        base = self.convolution_normalization_activation(base, 32, (3, 3), (1, 1), "valid")
        base = self.convolution_normalization_activation(base, 64, (3, 3), (1, 1), "same")

        branch_0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(base)
        branch_1 = self.convolution_normalization_activation(base, 96, (3, 3), (2, 2), "valid")
        
        base = concatenate([branch_0, branch_1])

        branch_0 = self.convolution_normalization_activation(base, 64, (1, 1), (1, 1), "same")
        branch_0 = self.convolution_normalization_activation(branch_0, 96, (3, 3), (1, 1), "valid")

        branch_1 = self.convolution_normalization_activation(base, 64, (1, 1), (1, 1), "same")
        branch_1 = self.convolution_normalization_activation(branch_1, 64, (7, 1), (1, 1), "same")
        branch_1 = self.convolution_normalization_activation(branch_1, 64, (1, 7), (1, 1), "same")
        branch_1 = self.convolution_normalization_activation(branch_1, 96, (3, 3), (1, 1), "valid")

        base = concatenate([branch_0, branch_1])

        branch_0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(base)
        branch_1 = self.convolution_normalization_activation(base, 192, (3, 3), (2, 2), "valid")
        
        base = concatenate([branch_0, branch_1])

        return base

    def inception_network(self, input):
        net = self.inception_stem(input)

        for _ in range(4):
            net = self.inception_block_A(net)
        
        net = self.reduction_block_A(net)

        for _ in range(7):
            net = self.inception_block_B(net)

        net = self.reduction_block_B(net)

        for _ in range(3):
            net = self.inception_block_C(net)
        
        net = AveragePooling2D(pool_size=(8, 8), padding="valid")(net)
        net = Dropout(0.2)(net)
        net = Flatten()(net)

        return net

    def build_model(self):
        inputs = Input(self.input_shape)

        net = self.inception_network(inputs)

        outputs = Dense(units=self.n_outputs, activation='softmax')(net)

        model = Model(inputs, outputs)

        print("INCEPTIONV4 ARCHITECTURE")

        model.summary()

        return model
