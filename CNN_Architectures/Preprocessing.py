
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Preprocessing(object):
	def __init__(self, file_path, input_shape, batch_size):
		self.reshape = (input_shape[0], input_shape[1])
		self.file_path = file_path
		self.batch_size = batch_size

	def preprocess_input(self, input):
		input = np.divide(input, 255.0)
		# input = np.subtract(input, 0.5)
		# input = np.multiply(input, 2.0)
		
		return input

	def train_val_test(self):
		self.data_gen = ImageDataGenerator(preprocessing_function=self.preprocess_input)

		data = self.data_gen.flow_from_directory(self.file_path, target_size=self.reshape, batch_size=self.batch_size, shuffle=True, interpolation="bicubic")

		X, Y = data.next()

		X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, shuffle=True, stratify=Y)

		print("X TRAIN: ", X_train.shape)
		print("Y TRAIN: ", Y_train.shape)
		print("X VAL: ", X_val.shape)
		print("Y VAL: ", Y_val.shape)

		return X_train, X_val, Y_train, Y_val
