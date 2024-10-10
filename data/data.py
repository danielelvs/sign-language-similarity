import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import re

from utils.logs import Logs


class Data(Dataset):

	def __init__(self, dataset_name='minds', image_method=None):
		self.dataset_name = dataset_name
		self.dataset = self.get_dataset()
		self.signs = self.get_signs(self.dataset)
		self.dataframe = self.prepare_data(self.dataset)
		self.categories = list(self.dataframe["category"].unique())
		self.persons = list(self.dataframe["person"].unique())
		self.image_method = image_method
		self.transform = transforms.ToTensor()  # Definir a transformação para tensor


	def __len__(self):
		return len(self.dataframe)


	def __getitem__(self, index):
		dataframe = self.dataframe.iloc[index]
		x, y, category, person = dataframe["x"], dataframe["y"], dataframe["category"], dataframe["person"]

		image = self.image_method().transform(x, y)
		image = Image.fromarray(np.uint8(image * 255)).convert('RGB')

		return self.transform(image).type(torch.float32), torch.tensor(self.categories.index(category), dtype=torch.int64), torch.tensor(self.persons.index(person), dtype=torch.int64)


	def get_dataset(self):
		try:
			dataset_file = f"libras_{self.dataset_name}_openpose.csv"
			dataset_path = os.path.join(f"datasets/{self.dataset_name}", dataset_file)
			return pd.read_csv(dataset_path, low_memory=True)
		except FileNotFoundError as e:
			Logs(logging.ERROR, f"Dataset {self.dataset_name} not found. Error: {e}")
			return None


	def get_features(self):
		features = []

		for index in range(self.dataframe.shape[0]):
			image, label, person = self[index]
			features.append({"X": image, "y": label, "person": person})

		return pd.DataFrame(features)


	def get_signs(self, df):
		signs = list(df.columns)
		signs = [s for s in signs if s.endswith("_x") or s.endswith("_y") or s.endswith("_z")]
		excluded_body_landmarks = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]
		excluded_body_landmarks = tuple([f"pose_{i}" for i in excluded_body_landmarks])
		unwanted_pose_columns = [i for i in list(signs) if i.startswith(excluded_body_landmarks)]
		signs = [s for s in signs if s not in unwanted_pose_columns]
		return signs


	def prepare_data(self, df):
		if (self.dataset_name == "minds"):
			if "person" not in df.columns:
				df["person"] = df["video_name"].apply(lambda i: int(re.findall(r".*Sinalizador(\d+)-.+.mp4", i)[0]))

		columns = ["category", "video_name", "person", "frame"] + self.signs
		df = df[columns]
		videos = df["video_name"].unique()
		data = []

		for video in videos:
			df_video = df[df["video_name"] == video].sort_values("frame")
			category = df_video.iloc[0]["category"]
			person = df_video.iloc[0]["person"]

			df_video = df_video.drop(["category", "video_name", "frame"], axis=1)
			x = self.get_axis_df(df_video, "x")
			y = self.get_axis_df(df_video, "y")

			x = x.T.to_numpy()
			y = y.T.to_numpy()

			x = self.normalize_axis(x)
			y = self.normalize_axis(y)

			data.append({
				"x": x,
				"y": y,
				"video_name": video,
				"category": category,
				"person": person
			})

		return pd.DataFrame.from_dict(data)


	def get_axis_df(self, df, axis):
		return df[[c for c in self.signs if c.endswith(axis)]]


	def normalize_axis(self, axis):
		axis[axis < 0] = 0
		axis[axis > 1] = 1
		return axis
