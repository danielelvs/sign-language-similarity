import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import re
import matplotlib.pyplot as plt
from collections import Counter

from utils.logs import Logs


class Data(Dataset):
	def __init__(self, dataset_name='minds', image_method=None, transform=None):
		self.dataset_name = dataset_name
		self.dataset = self.get_dataset()
		self.signs = self.get_signs(self.dataset)
		self.dataframe = self.prepare_data(self.dataset)
		self.categories = list(self.dataframe["category"].unique())
		self.persons = list(self.dataframe["person"].unique())
		self.image_method = image_method
		self.transform = transform


	def __getitem__(self, index):
		dataframe = self.dataframe.iloc[index]
		x, y, category, person = dataframe["x"], dataframe["y"], dataframe["category"], dataframe["person"]

		image = self.image_method().transform(x, y)
		image = Image.fromarray(np.uint8(image * 255)).convert('RGB')

		if self.transform:
			image = self.transform(image)

		return image, torch.tensor(self.categories.index(category), dtype=torch.int64), torch.tensor(self.persons.index(person), dtype=torch.int64)


	def __len__(self):
		return len(self.dataframe)


	def get_dataset(self):
		try:
			dataset_file = f"libras_{self.dataset_name}_openpose.csv"
			dataset_path = os.path.join(f"datasets/{self.dataset_name}", dataset_file)
			return pd.read_csv(dataset_path, low_memory=True)
		except FileNotFoundError as e:
			Logs(logging.ERROR, f"Dataset {self.dataset_name} not found. Error: {e}")
			return None


	def get_features(self):
		# total_image_size = self.image_size[0] * self.image_size[1] * 3  # Multiplicando por 3 para RGB

		_X, _y, _p = [], [], []
		# np.empty((0, total_image_size)), np.empty((0, )), np.empty((0, ))

		for index in range(self.dataframe.shape[0]):
			_image, _label, _person = self[index]

			_image = _image.cpu().numpy()
			_label = _label.cpu().numpy()
			_person = _person.cpu().numpy()

			_X.append(_image)
			_y.append(_label)
			_p.append(_person)

			# X = np.append(X, [image], axis=0)
			# y = np.append(y, [label], axis=0)
			# p = np.append(p, [person], axis=0)


		X = np.stack(_X)
		y = np.stack(_y)
		p = np.stack(_p)
		print(X.shape, y.shape, p.shape)

		return X, y, p


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


	def split(self, X, y):
		# X = np.load(f"{self.path}/X.npy", allow_pickle=True)
		# y = np.load(f"{self.path}/y.npy", allow_pickle=True)
		H = W = int(np.sqrt(X.shape[1]))
		X = X.reshape(-1, H, W)

		# print(f"#samples={X.shape[0]} min/max={X.min()}/{X.max()}")
		# print(f"image size={H}x{W}")
		# print(f"#classes={len(np.unique(y))}")

        # #----------------------------------------------

		classes = np.unique(y)
		num_classes = len(classes)

		# Split size
		num_classes_test = 5
		num_classes_val = 5
		num_classes_train = num_classes - num_classes_test - num_classes_val

		# Dataset
		X = X.astype(np.float32)
		y = y.astype(int)

		# Shuffle dataset
		indices = np.arange(len(y))
		np.random.shuffle(indices)
		X = X[indices]
		y = y[indices]

		# Split classes
		train_classes = np.random.choice(classes, num_classes_train, replace=False) # select random classes for training
		train_classes.sort()
		rem_classes = np.setdiff1d(classes, train_classes) # remaining classes
		val_classes = np.random.choice(rem_classes, num_classes_val, replace=False) # select random classes for validation
		val_classes.sort()
		test_classes = np.setdiff1d(rem_classes, val_classes) # remaining classes
		# print(f"train_classes={train_classes}")
		# print(f"val_classes={val_classes}")
		# print(f"test_classes={test_classes}")

		train_idx = np.isin(y, train_classes)
		val_idx = np.isin(y, val_classes)
		test_idx = np.isin(y, test_classes)

		X_train, y_train = X[train_idx], y[train_idx]
		X_val, y_val = X[val_idx], y[val_idx]
		X_test, y_test = X[test_idx], y[test_idx]

		# print(f"X_train={X_train.shape}")
		# print(f"y_train={y_train.shape}")
		# print(f"X_val={X_val.shape}")
		# print(f"y_val={y_val.shape}")
		# print(f"X_test={X_test.shape}")
		# print(f"y_test={y_test.shape}")

		# np.save(os.path.join(self.path, "X_train.npy"), X_train)
		# np.save(os.path.join(self.path, "y_train.npy"), y_train)
		# np.save(os.path.join(self.path, "X_val.npy"), X_val)
		# np.save(os.path.join(self.path, "y_val.npy"), y_val)
		# np.save(os.path.join(self.path, "X_test.npy"), X_test)
		# np.save(os.path.join(self.path, "y_test.npy"), y_test)

		return X_train, y_train, X_val, y_val, X_test, y_test


	def samples(self, y_train, y_val, y_test):
		"""
		Plots the distribution of training, validation, and test samples by class.
		This method creates a bar plot showing the count of samples for each class
		in the training, validation, and test datasets. The x-axis represents the
		class labels, and the y-axis represents the count of samples. The plot
		includes a legend to differentiate between the training, validation, and
		test datasets.
		"""

		counter_train, counter_val, counter_test = Counter(y_train), Counter(y_val), Counter(y_test)
		plt.bar(counter_train.keys(), counter_train.values(), label='train')
		plt.bar(counter_val.keys(), counter_val.values(), label='val')
		plt.bar(counter_test.keys(), counter_test.values(), label='test')
		plt.xlabel('Class')
		plt.ylabel('Count')
		plt.title('Validation samples distribution by class')
		plt.legend()
		plt.show()
