import matplotlib.pyplot as plt
import numpy as np
import torch


class Train:
	"""
	A class used to encapsulate the training process of a model.
	"""

	def __init__():
		pass

	def execution(model, train_dataloader, val_dataloader, num_epochs, criterion, optimizer, device, evaluator, k_shot):

		model.train()
		best_loss = float("inf")
		best_loss_history = []
		loss_history = []

		best_epoch = -1
		best_accuracy = -1
		accuracy_history = []

		iteration = 0

		for epoch in range(num_epochs):
			model.train()

			for idx, (image1, image2, label) in enumerate(train_dataloader, 0):
				image1, image2, label = image1.to(device), image2.to(device), label.to(device)
				optimizer.zero_grad()
				output1, output2 = model(image1, image2)
				loss_contrastive = criterion(output1, output2, label)
				loss_contrastive.backward()
				optimizer.step()

				loss_value = loss_contrastive.item()
				if loss_value < best_loss:
					best_loss = loss_value
				best_loss_history.append(best_loss)
				loss_history.append(loss_value)

				if idx % 10 == 0:
					loss_mean = np.mean(loss_history[-10:])
					print(f"Epoch {epoch} Loss {loss_mean:.4f} (Best Epoch {best_epoch} Best Accuracy {best_accuracy:3f})")

				iteration += 1

			accuracy = evaluator.eval(model, val_dataloader, k_shot)
			accuracy_history.append(accuracy)

			if accuracy > best_accuracy:
				best_accuracy = accuracy
				best_epoch = epoch

				if not os.path.exists("../checkpoints"):
					os.mkdir(category_path)

				torch.save(model.state_dict(), "checkpoints/best-model.pth")

		return best_epoch, best_accuracy, loss_history, accuracy_history


	def chart(num_epochs, loss_history, accuracy_history):
		"""
		Plots the training loss and accuracy over epochs.
		Args:
			loss_history (list or np.ndarray): A list or array containing the training loss values.
			accuracy_history (list or np.ndarray): A list or array containing the accuracy values.
		Returns:
			None: This function does not return any value. It displays a plot.
		"""

		iterations_val = np.linspace(0, len(loss_history) - 1, num_epochs, dtype=int)

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
