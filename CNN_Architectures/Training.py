from tensorflow.keras.callbacks import EarlyStopping

class TrainEvaluate(object):
	def __init__(self, model, path, X_train, Y_train, X_val=None, Y_val=None, X_test=None, Y_test=None):
		self.model = model
		self.path = path
		self.X_train = X_train
		self.Y_train = Y_train
		self.X_val = X_val
		self.Y_val = Y_val
		self.X_test = X_test
		self.Y_test = Y_test

	def train_evaluate(self):
		self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

		es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20, restore_best_weights=True)

		self.model.fit(self.X_train, self.Y_train, epochs=1000, batch_size=512, callbacks=[es], validation_data=(self.X_val, self.Y_val), shuffle=True)

		_, self.train_accuracy = self.model.evaluate(self.X_train, self.Y_train)
		_, self.validation_accuracy = self.model.evaluate(self.X_val, self.Y_val)

		print("Train Accuracy: ", self.train_accuracy)
		print("Validation Accuracy: ", self.validation_accuracy)

		tf.saved_model.save(self.model, "baseline_alexnet")
