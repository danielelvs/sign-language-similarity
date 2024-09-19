import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
import torch

class Evaluate():

	def __init__(self, labels, k_shot=1, num_runs=None, device=None):

		if num_runs is None:
			num_runs = len(labels) // len(np.unique(labels)) # avg. samples per class

		self.device = device if device is not None else "cpu"
		self.k_shot = k_shot
		self.num_runs = num_runs

		# Setup support and query
		self.support_idxs = []
		self.query_idxs = []
		for _ in range(num_runs):
			support_idxs, query_idxs = self.support(labels, k_shot)
			self.support_idxs.append(support_idxs)
			self.query_idxs.append(query_idxs)

	def support(self, labels, k):
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

	def eval(self, model, dataloader, k=3):
		assert 1 <= k <= self.k_shot, "k must be less than or equal to k_eval"

		# Inference
		outputs = []
		labels = []
		model.eval()
		with torch.no_grad():
				for _, (image, label) in enumerate(dataloader, 0):
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
