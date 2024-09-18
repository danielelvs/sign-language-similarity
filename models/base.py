from abc import ABC, abstractmethod

class BaseModel(ABC):
	"""
	BaseModel is an abstract base class that defines the structure for models.

	Attributes:
			name (str): The name of the model.
			image_size (tuple): The size of the input images.
	"""

	name: str
	image_size: tuple

	def __init__(self, num_classes: int):
		"""
		Initializes the model with the specified number of classes.

		Args:
				num_classes (int): The number of classes for the classification task.
		"""
		self.num_classes = num_classes

	@abstractmethod
	def get_model(self):
		"""
		Retrieve the model instance.

		This method should be overridden by subclasses to return the specific model
		instance that they represent.
		"""
		pass

	@abstractmethod
	def get_fc_layer(self):
		"""
		Returns the fully connected layer of the model.

		This method should be overridden by subclasses to provide the specific
		implementation of the fully connected layer.
		"""
		pass

	@abstractmethod
	def get_transformers(self):
		"""
		Retrieves the transformers used in the model.

		This method should be overridden by subclasses to return the specific
		transformers required for the model's operation.

		"""
		pass

	@staticmethod
	def get_by_name(name):
		"""
		Retrieve a subclass of BaseModel by its name.

		Args:
			name (str): The name of the subclass to retrieve.

		Returns:
			type: The subclass of BaseModel with the specified name, or None if no such subclass exists.
		"""
		classes = { cls.name: cls for cls in BaseModel.__subclasses__() }
		return classes.get(name)
