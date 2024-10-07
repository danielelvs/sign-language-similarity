from abc import ABC, abstractmethod


class BaseRepresentation(ABC):
    name: str


    @abstractmethod
    def transform(self, x, y, z):
        pass


    @staticmethod
    def get_by_name(name):
        classes = {cls.name: cls for cls in BaseRepresentation.__subclasses__()}
        return classes.get(name)
