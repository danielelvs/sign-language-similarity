from matplotlib import pyplot as plt
import numpy as np
import torch
from steps import evaluate

class Train:
	def __init__(self, model, train_dataloader, val_dataloader, num_epochs, criterion, optimizer, device):

		self.model = model
		self.train_dataloader = train_dataloader
		self.val_dataloader = val_dataloader
		self.num_epochs = num_epochs
		self.criterion = criterion
		self.optimizer = optimizer
		self.device = device

	def train(self):
		self.model.train()
		best_loss = float("inf")
		best_loss_history = []
		loss_history = []

		best_epoch = -1
		best_accuracy = -1
		accuracy_history = []

		iteration = 0

		for epoch in range(self.num_epochs):
			self.model.train()

			for idx, (image1, image2, label) in enumerate(self.train_dataloader, 0):
				image1, image2, label = image1.to(self.device), image2.to(self.device), label.to(self.device)
				self.optimizer.zero_grad()
				output1, output2 = self.model(image1, image2)
				loss_contrastive = self.criterion(output1, output2, label)
				loss_contrastive.backward()
				self.optimizer.step()

				loss_value = loss_contrastive.item()
				if loss_value < best_loss:
					best_loss = loss_value
				best_loss_history.append(best_loss)
				loss_history.append(loss_value)

				if idx % 10 == 0:
					loss_mean = np.mean(loss_history[-10:])
					print(f"Epoch {epoch} Loss {loss_mean:.4f} (Best Epoch {best_epoch} Best Accuracy {best_accuracy:3f})")

				iteration += 1

			accuracy = evaluate.eval(self.model, self.val_dataloader, k=1)
			accuracy_history.append(accuracy)

			if accuracy > best_accuracy:
				best_accuracy = accuracy
				best_epoch = epoch
				torch.save(self.model.state_dict(), "best-model.pth")

		return best_epoch, best_accuracy, best_loss_history, loss_history, accuracy_history

	def chart(self, loss_history, accuracy_history):
		iterations_val = np.linspace(0, len(loss_history) - 1, self.num_epochs, dtype=int)

		_, ax1 = plt.subplots(figsize=(10, 5))
		ax1.plot(loss_history, "b-", label="Training Loss")
		ax1.set_ylabel("Loss")
		ax1.set_xlabel("Iteration")

		ax2 = ax1.twinx()
		ax2.plot(iterations_val, accuracy_history, "-", label="Accuracy", color="orange")
		ax2.set_ylabel("Accuracy")

		handle1, label1 = ax1.get_legend_handles_labels()
		handle2, label2 = ax2.get_legend_handles_labels()

		handles = handle1 + handle2
		labels = label1 + label2
		ax1.legend(handles, labels, loc='upper right')
		plt.title("Training curve")
		plt.show()
