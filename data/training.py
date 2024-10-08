import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
import torch
from torch.utils.data import Dataset


class TrainingData(Dataset):

    def __init__(self, X, y, size=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.size = size if size is not None else X.shape[0]


    def __getitem__(self):
        idx1 = np.random.randint(0, len(self.X))
        image1 = torch.tensor(self.X[idx1:idx1+1], dtype=torch.float32)
        label1 = self.y[idx1]

        should_match = np.random.choice([True, False])
        mask_candidates = (self.y == label1) if should_match else (self.y != label1)
        mask_candidates[idx1] = False
        idx_candidates = np.where(mask_candidates)[0]
        idx2 = np.random.choice(idx_candidates)

        image2 = torch.tensor(self.X[idx2:idx2+1], dtype=torch.float32)
        #label2 = self.y[idx2]

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.tensor(1 - should_match, dtype=torch.float32)


    def __len__(self):
        return self.size


    def split(self):
        X = np.load(f"{self.path}/X.npy", allow_pickle=True)
        y = np.load(f"{self.path}/y.npy", allow_pickle=True)
        H = W = int(np.sqrt(X.shape[1]))
        X = X.reshape(-1, H, W)

        print(f"#samples={X.shape[0]} min/max={X.min()}/{X.max()}")
        print(f"image size={H}x{W}")
        print(f"#classes={len(np.unique(y))}")

        #----------------------------------------------

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

        print(f"train_classes={train_classes}")
        print(f"val_classes={val_classes}")
        print(f"test_classes={test_classes}")

        train_idx = np.isin(y, train_classes)
        val_idx = np.isin(y, val_classes)
        test_idx = np.isin(y, test_classes)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

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
