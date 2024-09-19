import os
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter


class Dataset:

  @property
  def X(self):
    return self._X

  @X.setter
  def X(self, value):
    self._X = value

  @property
  def y(self):
    return self._y

  @y.setter
  def y(self, value):
    self._y = value


  def __init__(self, folder="data", dataset="OlivettiFaces"):
    self.folder = folder
    self.dataset = dataset
    self.path = os.path.join(self.folder, self.dataset)


  def load(self):
    X_file_path = f"{self.path}/X.npy"
    y_file_path = f"{self.path}/y.npy"

    self.X = np.load(X_file_path, allow_pickle=True)
    self.y = np.load(y_file_path, allow_pickle=True)

    H = W = int(np.sqrt(self.X.shape[1]))
    self.X = self.X.reshape(-1, H, W)

    print(f"#samples={self.X.shape[0]} min/max={self.X.min()}/{self.X.max()}")
    print(f"image size={H}x{W}")
    print(f"#classes={len(np.unique(self.y))}")

    return self.X, self.y


  def split(self):
    classes = np.unique(self.y) # identify unique classes
    num_classes = len(classes)

    # Split size
    num_classes_test = 5
    num_classes_val = 5
    num_classes_train = num_classes - num_classes_test - num_classes_val

    # Dataset
    self.X = self.X.astype(np.float32)
    self.y = self.y.astype(int)

    # Shuffle dataset
    indices = np.arange(len(self.y))
    np.random.shuffle(indices)
    self.X = self.X[indices]
    self.y = self.y[indices]

    # Split classes
    train_classes = np.random.choice(classes, num_classes_train, replace=False) # select random classes for training
    train_classes.sort()
    rem_classes = np.setdiff1d(classes, train_classes) # remaining classes

    val_classes = np.random.choice(rem_classes, num_classes_val, replace=False) # select random classes for validation
    val_classes.sort()
    test_classes = np.setdiff1d(rem_classes, val_classes) # remaining classes

    print(f"train_classes={train_classes}")
    print(f"val_classes={val_classes}")
    print(f"test_classes={test_classes}")

    train_idx = np.isin(self.y, train_classes)
    val_idx = np.isin(self.y, val_classes)
    test_idx = np.isin(self.y, test_classes)

    X_train, y_train = self.X[train_idx], self.y[train_idx]
    X_val, y_val = self.X[val_idx], self.y[val_idx]
    X_test, y_test = self.X[test_idx], self.y[test_idx]

    print(f"X_train={X_train.shape}")
    print(f"y_train={y_train.shape}")
    print(f"X_val={X_val.shape}")
    print(f"y_val={y_val.shape}")
    print(f"X_test={X_test.shape}")
    print(f"y_test={y_test.shape}")

    np.save(os.path.join(self.path, "X_train.npy"), X_train)
    np.save(os.path.join(self.path, "y_train.npy"), y_train)
    np.save(os.path.join(self.path, "X_val.npy"), X_val)
    np.save(os.path.join(self.path, "y_val.npy"), y_val)
    np.save(os.path.join(self.path, "X_test.npy"), X_test)
    np.save(os.path.join(self.path, "y_test.npy"), y_test)

    # Plot validation samples distribution by class
    counter_train, counter_val, counter_test = Counter(y_train), Counter(y_val), Counter(y_test)
    plt.bar(counter_train.keys(), counter_train.values(), label='train')
    plt.bar(counter_val.keys(), counter_val.values(), label='val')
    plt.bar(counter_test.keys(), counter_test.values(), label='test')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Validation samples distribution by class')
    plt.legend()
    plt.show()
