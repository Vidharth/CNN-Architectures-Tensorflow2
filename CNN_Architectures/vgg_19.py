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


class VGG19(Model):
	def __init__(self, classes):
		super(VGG19, self).__init__()


        