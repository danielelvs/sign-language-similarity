import time
import numpy as np
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset

class Transform(Dataset):

  def __init__(self, data, dataset_name='ufop', person_in=[], person_out=[], image_method=None):
    self.data = data
    self.dataset_name = dataset_name
    self.categories = list(data["category"].unique())
    self.persons = list(data["person"].unique())

    self.person_in = person_in
    self.person_out = person_out
    self.image_method = image_method

    self.signs = self.get_signs(data)
    self.dataframe = self.prepare_data(data)


  def __getitem__(self, index):
    dataset_path = os.path.join("data", self.dataset_name)
    if not os.path.exists(dataset_path):
      os.mkdir(dataset_path)

    print(dataset_path)

    person = dataframe["person"]
    person_path = os.path.join(dataset_path, person)
    if not os.path.exists(person_path):
      os.mkdir(person_path)

    print(person_path)

    category = dataframe["category"]
    category_path = os.path.join(person_path, category)
    if not os.path.exists(category_path):
      os.mkdir(category_path)

    print(category_path)

    dataframe = self.dataframe.iloc[index]
    x = dataframe["x"]
    y = dataframe["y"]
    z = dataframe["z"]

    image = self.landmarks_to_image(x, y, z)
    np.save(f"data/{self.dataset_name}/{person}/{category}.npy", image)

    image = Image.fromarray(np.uint8(image * 255)).convert('RGB')
    image.save(f"data/{self.dataset_name}/{person}/{category}.png")

    return image, torch.tensor(self.categories.index(category), dtype=torch.int64), torch.tensor(self.persons.index(person), dtype=torch.int64)


  def __len__(self):
    return len(self.dataframe)


  def prepare_data(self, df):
    columns = ["category", "video_name", "person", "frame"] + self.signs
    df = df[columns]
    videos = df["video_name"].unique()
    data = []

    for video in videos:
      video_name = df[df["video_name"] == video].sort_values("frame")
      category = video_name.iloc[0]["category"]
      person = video_name.iloc[0]["person"]

      if len(self.person_in) > 0:
        if person not in self.person_in:
          continue

      if len(self.person_out) > 0:
        if person in self.person_out:
          continue

      video_name = video_name.drop(["category", "video_name", "frame"], axis=1)
      x = self.get_axis_df(video_name, "x")
      y = self.get_axis_df(video_name, "y")
      z = self.get_axis_df(video_name, "z")

      x = x.T.to_numpy()
      y = y.T.to_numpy()
      z = z.T.to_numpy()

      x = self.normalize_axis(x)
      y = self.normalize_axis(y)
      z = self.normalize_axis(z)

      data.append({
        "video_name": video,
        "x": x,
        "y": y,
        "z": z,
        "category": category
      })

    return pd.DataFrame.from_dict(data)


  def get_signs(self, df):
    signs = list(df.columns)
    signs = [s for s in signs if s.endswith("_x") or s.endswith("_y") or s.endswith("_z")]
    excluded_body_landmarks = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]
    excluded_body_landmarks = tuple([f"pose_{i}" for i in excluded_body_landmarks])
    unwanted_pose_columns = [i for i in list(signs) if i.startswith(excluded_body_landmarks)]
    signs = [s for s in signs if s not in unwanted_pose_columns]
    return signs


  def get_axis_df(self, df, axis):
    return df[[c for c in self.signs if c.endswith(axis)]]


  def normalize_axis(self, axis):
    axis[axis < 0] = 0
    axis[axis > 1] = 1
    return axis


  def landmarks_to_image(self, x, y, z):
    start_time = time.time()
    image = self.skeleton_dml(x, y, z)
    end_time = time.time()
    print(f"landmarks_to_image duration: {end_time - start_time}")
    return image


  def skeleton_dml(self, x, y, z):
    n = 3
    width = x.shape[1]

    if width % n != 0:
      extra_cols = width % n
      x = x[:, : width - extra_cols]
      y = y[:, : width - extra_cols]

    x = np.reshape(x, (x.shape[0], -1, n))
    y = np.reshape(y, (y.shape[0], -1, n))

    image = np.concatenate([x, y], axis=1)

    return image

