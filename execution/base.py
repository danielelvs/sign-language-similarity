from abc import ABC, abstractmethod

from model import Model
from utils.args import Args


class BaseExecution(ABC):
    step: str

    def __init__(self):
        self.args = Args()
        # self.model = Model.get_by_name(self.args.network)(self.args.num_classes)


    def build(self):

        pass
        # self.model = Sequential()
        # self.model.add(Dense(1, input_dim=1))


    def train(self, x, y):
        pass
        # self.model.compile(optimizer='adam', loss='mean_squared_error')
        # self.model.fit(x, y, epochs=100)


    def predict(self, x):
        pass
        # return self.model.predict(x)


    def save(self, path):
        pass
        # self.model.save(path)


    def load(self, path):
        pass
        # self.model = load_model(path)


    @staticmethod
    def get_step(name):
        classes = {cls.name: cls for cls in BaseExecution.__subclasses__()}
        return classes.get(name)
