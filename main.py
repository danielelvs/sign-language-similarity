import random
import numpy as np
import torch
from data.data import Data
from loaders.eval import EvalDataset
from loaders.train import TrainDataset
from models.siamese import Siamese
from steps.evaluator import Evaluator
from steps.train import Train
from steps.transform import Transform
from utils.loss import ContrastiveLoss
from utils.params import Params
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import time
import numpy as np
import torch
from data.data import Data
from loaders.eval import EvalDataset
from loaders.train import TrainDataset
from models.siamese import Siamese
from steps.evaluator import Evaluator
from steps.train import Train
from utils.loss import ContrastiveLoss
from utils.params import Params
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim


def main():
  dataset_name = "ufop" #vir por params
  data_name = f"libras_{dataset_name}_openpose.csv"

  data_path = os.path.join("data", dataset_name)
  if not os.path.exists(data_path):
    os.mkdir(data_path)

  data_csv = pd.read_csv(f'{data_path}/{data_name}')

  _transformation = Transform(data_csv)
  print(_transformation)

  # ###############

  # params = Params().args

  # SEED = params.random_seed
  # torch.manual_seed(SEED)
  # np.random.seed(SEED)
  # random.seed(SEED)

  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # feat_dim = 128 #params.feat_dim
  # learning_rate = params.learning_rate
  # epochs = params.epochs
  # batch_size = params.batch_size
  # shuffle_data = params.shuffle_data
  # num_workers = params.num_workers
  # num_runs = params.num_runs
  # feat_dim = params.feat_dim
  # k_shot = params.k_shot
  # model_name = params.model_name

  # data = Data()
  # X_train, y_train, X_val, y_val, X_test, y_test = data.split()
  # data.samples(y_train=y_train, y_val=y_val, y_test=y_test)

  # transformation = transforms.Compose([transforms.Resize((100, 100))])

  # train_dataset = TrainDataset(X=X_train, y=y_train, transform=transformation)
  # train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size)

  # val_dataset = EvalDataset(X=X_val, y=y_val, transform=transformation)
  # val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)

  # test_dataset = EvalDataset(X=X_test, y=y_test, transform=transformation)
  # test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)

  # model = Siamese(feat_dim=feat_dim).to(device)
  # criterion = ContrastiveLoss()
  # optimizer = optim.Adam(model.parameters(), lr=learning_rate)


  # # treinamento
  # train_evaluator = Evaluator(labels=y_val, k_shot=k_shot, num_runs=num_runs, device=device)
  # best_epoch, best_accuracy, loss_history, accuracy_history = Train.execution(
  #   model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, num_epochs=epochs, criterion=criterion, optimizer=optimizer, device=device, evaluator=train_evaluator, k_shot=k_shot)
  # print(f"Best Epoch {best_epoch} Best Accuracy {best_accuracy:3f}")
  # Train.chart(epochs, loss_history, accuracy_history)


  # # avaliação
  # model.load_state_dict(torch.load("checkpoints/best-model.pth", weights_only=True))
  # evaluator = Evaluator(labels=y_test, k_shot=k_shot, num_runs=num_runs, device=device)
  # accuracy = evaluator.eval(model=model, dataloader=test_dataloader, k=k_shot)
  # print(f"Accuracy: {accuracy}")

  # avaliação
  model.load_state_dict(torch.load("checkpoints/best-model.pth", weights_only=True))
  evaluator = Evaluator(labels=y_test, k_shot=k_shot, num_runs=num_runs, device=device)
  accuracy = evaluator.eval(model=model, dataloader=test_dataloader, k=k_shot)
  print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
  main()
