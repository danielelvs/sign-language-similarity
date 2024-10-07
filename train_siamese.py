import logging
import os
import re
import pandas as pd
from execution.base import Model
from utils.logs import Logs
from utils.args import Params


class TrainSiamese:

    def __init__(self):
        super(TrainSiamese, self).__init__()
        self.args = Params()
        self.model = Model()


    def exec(self):
        try:
            dataset_name = self.args.dataset_name
            Logs(logging.INFO, f"Loading dataset {dataset_name}...")
            dataset = self.get_dataset(dataset_name)

            if (dataset_name == "minds"):
                if "person" not in dataset.columns:
                    dataset["person"] = dataset["video_name"].apply(lambda i: int(re.findall(r".*Sinalizador(\d+)-.+.mp4", i)[0]))

            # self.model.train(dataset)

        except Exception as e:
            Logs(logging.ERROR, f"Error: {e}")


    def get_dataset(self, dataset_name):
        try:
            dataset_file = f"libras_{dataset_name}_openpose.csv"
            dataset_path = os.path.join(f"datasets/{dataset_name}", dataset_file)
            return pd.read_csv(dataset_path, low_memory=True)
        except FileNotFoundError:
            Logs(logging.ERROR, f"Dataset {dataset_name} not found.")
            return None
