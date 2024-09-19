# import random
# from matplotlib import transforms
# import numpy as np
# import torch
from data.dataset import Dataset
# from loaders.evaluate import EvaluateLoader
# from loaders.train import TrainLoader
# from models.siamese import Siamese
# from torch.utils.data import DataLoader
# from steps.evaluate import Evaluate
# from steps.train import Train
# from utils.loss import ContrastiveLoss
# from torch import optim

# SEED = 42
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#   torch.manual_seed(SEED)
#   np.random.seed(SEED)
#   random.seed(SEED)

#   feat_dim = 128
#   learning_rate = 0.0001
#   num_epochs = 1000
#   batch_size = 32
#   num_workers = 2 # for parallel processing
#   num_runs = 100 # number of evaluations for validation/test

#   transformation = transforms.Compose([transforms.Resize((100, 100))])

#   train_dataset = TrainLoader(X=X_train, y=y_train, transform=transformation)
#   train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size)

#   val_dataset = EvaluateLoader(X=X_val, y=y_val, transform=transformation)
#   val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)

#   test_dataset = EvaluateLoader(X=X_test, y=y_test, transform=transformation)
#   test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)

#   model = Siamese(feat_dim=feat_dim).to(device)
#   criterion = ContrastiveLoss()
#   optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#   # evaluator = Evaluate(labels=y_val, k_shot=1, num_runs=num_runs, device=device)

#   # treinamento
#   training = Train()
#   best_epoch, best_accuracy, best_loss_history, loss_history, accuracy_history = training(model, train_dataloader, val_dataloader, num_epochs=num_epochs, criterion=criterion, optimizer=optimizer, device=device)
#   print(f"Best Epoch {best_epoch} Best Accuracy {best_accuracy:3f}")
#   training.chart(loss_history, accuracy_history, num_epochs)

#   # avaliação'
#   model.load_state_dict(torch.load("best-model.pth", weights_only=True))
#   evaluator = Evaluate(labels=y_test, k_shot=1, num_runs=num_runs, device=device)
#   accuracy = evaluator.eval(model, test_dataloader, k=1)
#   print(f"Accuracy: {accuracy}")

def main():
  # 1. load params from yaml file
  # 2. load dataset

  dataset = Dataset().load()
  print(dataset)

if __name__ == "__main__":
  main()



# def main(hparams):
#     """
#     Main training routine specific for this project
#     :param hparams:
#     """
#     # ------------------------
#     # 1 INIT LIGHTNING MODEL
#     # ------------------------
#     model = Model(hparams)

#     # ------------------------
#     # 2 INIT TRAINER
#     # ------------------------
#     trainer = Trainer()

#     # ------------------------
#     # 3 START TRAINING
#     # ------------------------
#     trainer.fit(model)


# if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    # root_dir = os.path.dirname(os.path.realpath(__file__))
    # parent_parser = ArgumentParser(add_help=False)

    # each LightningModule defines arguments relevant to it
    # parser = Model.add_model_specific_args(parent_parser, root_dir)
    # hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    # main(hyperparams)
