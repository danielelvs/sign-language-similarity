from abc import ABC, abstractmethod


class BaseNetwork(ABC):
    name: str
    image_size: tuple


    def __init__(self, num_classes: int):
        self.num_classes = num_classes


    @abstractmethod
    def get_network(self):
        pass


    @abstractmethod
    def get_fc_layer(self):
        pass


    @abstractmethod
    def get_transformers(self):
        pass


    @staticmethod
    def get_by_name(name):
        classes = {cls.name: cls for cls in BaseNetwork.__subclasses__()}
        return classes.get(name)
