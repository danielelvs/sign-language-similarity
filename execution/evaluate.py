import logging
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
import torch

from utils.logs import Logs


class Evaluate:

	def __init__(self, labels, k_shot=1, num_runs=None, device=None):
		"""
		Initialize the evaluation setup with given labels and parameters.

		Args:
			labels (list or array-like): The labels for the dataset.
			k_shot (int, optional): The number of samples per class for the support set. Default is 1.
			num_runs (int, optional): The number of runs to perform. If None, it is set to the average number of samples per class. Default is None.
			device (str, optional): The device to use for computation (e.g., "cpu" or "cuda"). Default is None, which sets the device to "cpu".

		Attributes:
			device (str): The device used for computation.
			k_shot (int): The number of samples per class for the support set.
			num_runs (int): The number of runs to perform.
			support_idxs (list): A list of support set indices for each run.
			query_idxs (list): A list of query set indices for each run.
		"""

		if num_runs is None:
			num_runs = len(labels) // len(np.unique(labels)) # avg. samples per class

		self.device = device if device is not None else "cpu"
		self.k_shot = k_shot
		self.num_runs = num_runs

		# Setup support and query
		self.support_idxs = []
		self.query_idxs = []
		for _ in range(num_runs):
			support_idxs, query_idxs = self._split_into_query_support(labels, k_shot)
			self.support_idxs.append(support_idxs)
			self.query_idxs.append(query_idxs)


	def _split_into_query_support(self, labels, k):
		"""
		Splits the dataset into support and query sets based on the provided labels.

		Args:
			labels (array-like): Array of labels corresponding to the dataset.
			k (int): Number of samples to include in the support set for each unique label.

		Returns:
			tuple: Two lists containing the indices for the support set and the query set, respectively.
		"""
		# Get the unique labels
		labels_unique = np.unique(labels)

		# Initialize lists to hold the support and query indices
		support_idxs = []
		query_idxs = []

		# Sample k indices from each label/class
		for label in labels_unique:
			cls_idxs = np.where(labels == label)[0]

			# Support indices
			cls_support_idxs = resample(cls_idxs, replace=False, n_samples=k)

			# Determine the query indices (those not in the support set)
			cls_query_idxs = np.setdiff1d(cls_idxs, cls_support_idxs)

			support_idxs.extend(cls_support_idxs)
			query_idxs.extend(cls_query_idxs)

		return support_idxs, query_idxs

	def exec(self, model, dataloader, k=3) -> float:
		"""
		Evaluate the model using k-Nearest Neighbors (k-NN) classification.
		Args:
			model (torch.nn.Module): The model to be evaluated.
			dataloader (torch.utils.data.DataLoader): DataLoader providing the dataset for evaluation.
			k (int, optional): The number of neighbors to use for k-NN classification. Default is 3.
		Returns:
			float: The accuracy of the model on the evaluation dataset.
		Raises:
			AssertionError: If k is not between 1 and self.k_shot (inclusive).
		"""

		assert 1 <= k <= self.k_shot, Logs(level=logging.ERROR, msg="k must be less than or equal to k_eval")

		outputs = []
		labels = []
		model.eval()
		with torch.no_grad():
			for idx, (image, label) in enumerate(dataloader, 0):
				output = model.forward_once(image.to(self.device))
				outputs.append(output.cpu().numpy())
				labels.append(label.numpy())
		outputs = np.vstack(outputs)
		labels = np.concatenate(labels)

		# KNN
		hits = 0
		num_queries = 0
		neigh = KNeighborsClassifier(n_neighbors=k)
		for support_idxs, query_idxs in zip(self.support_idxs, self.query_idxs):
			neigh.fit(outputs[support_idxs], labels[support_idxs])
			labels_pred = neigh.predict(outputs[query_idxs])
			hits += np.sum(labels_pred == labels[query_idxs])
			num_queries += len(query_idxs)
		accuracy = hits / num_queries

		return accuracy
