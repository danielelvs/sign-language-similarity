import logging
import random
import numpy as np
import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data.data import Data
from data.evaluate import EvaluateData
from data.training import TrainingData
from execution.evaluate import Evaluate
from execution.trainining import Training
from model.base import BaseModel
from representations.base import BaseImageRepresentation
from utils.arguments import ArgumentsParser
from utils.logs import Logs
from utils.loss import ContrastiveLoss



def main():
    # load arguments
    args = ArgumentsParser().parse_args()

    seed = args.random_seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim = args.feat_dim
    learning_rate = args.learning_rate
    epochs = 1 #args.epochs
    batch_size = args.batch_size
    # shuffle_data = args.shuffle_data
    num_workers = args.num_workers
    num_runs = args.num_runs
    feat_dim = args.feat_dim
    k_shot = args.k_shot
    model_name = args.model_name
    dataset_name = 'ufop' #args.dataset_name
    image_representation = args.image_representation

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1. Get Image Representation
    Logs(level=logging.INFO, message=f"1. Getting {image_representation} image representation...")
    image_method = BaseImageRepresentation.get_type(image_representation)
    if image_method is None:
        Logs(level=logging.ERROR, message=f"Invalid image representation type.")
        raise ValueError("Invalid image representation type.")
    # print(image_method)

    # 2. Load Dataset
    Logs(level=logging.INFO, message=f"2. Loading dataset {dataset_name}...")
    data = Data(dataset_name=dataset_name, image_method=image_method)
    if data is None:
        Logs(level=logging.ERROR, message=f"Invalid dataset.")
        raise ValueError("Invalid dataset.")
    # print(data.dataset)


    # 3. Load Model
    Logs(level=logging.INFO, message=f"3. Loading model {model_name}...")
    if model_name == 'Siamese':
        param = feat_dim
    else:
        param = len(data["category"].unique())
    model = BaseModel.get_type(model_name)(param).to(device)
    if model is None:
        Logs(level=logging.ERROR, message=f"Invalid model type.")
        raise ValueError("Invalid model type.")
    # print(model)




    transform = transforms.Compose([
        model.image_size,
        transforms.ToTensor(),
    ])
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # 4. Split Dataset
    features = data.get_features()
    # X = features['X'].apply(lambda x: x.cpu())
    # y = features['y'].apply(lambda x: x.cpu())

    # X_shapes = [tensor.shape for tensor in X]
    # y_shapes = [tensor.shape for tensor in y]

    # print(X_shapes, y_shapes)

    # print(y.shape)  # Output: torch.Size([])
    # y_1d = y.squeeze(0)  # Convertendo para tensor 1-dimensional
    # print(y_1d.shape)  # Output: torch.Size([1])
    # print(y_1d)  # Output: tensor([5])


    # print(f"#classes={len(np.unique(y))}")
    # print(f"#samples={X[0].shape} min/max={X.min()}/{X.max()}")

    ############################# DATASETS #############################

    # treinamento
    # train_dataset = TrainingData(X=X_train, y=y_train, transform=transform)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size)


    # avaliação
    # eval_dataset = EvaluateData(X=X_val, y=y_val, transform=transform)
    # eval_dataloader = DataLoader(eval_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)

    # teste
    # test_dataset = EvaluateData(X=X_test, y=y_test, transform=transform)
    # test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)


    ############################# EXECUCOES #############################

    # treinamento
    # train_evaluator = Evaluate(labels=y_val, k_shot=k_shot, num_runs=num_runs, device=device)
    # best_epoch, best_accuracy, loss_history, accuracy_history = Training(
    # model=model, train_dataloader=train_dataloader, val_dataloader=eval_dataloader, num_epochs=epochs, criterion=criterion, optimizer=optimizer, device=device, evaluator=train_evaluator, k_shot=k_shot)
    # print(f"Best Epoch {best_epoch} Best Accuracy {best_accuracy:3f}")
    # Training.chart(epochs, loss_history, accuracy_history)

    # avaliação
    # model.load_state_dict(torch.load("checkpoints/best-model.pth", weights_only=True))
    # evaluator = Evaluate(labels=y_test, k_shot=k_shot, num_runs=num_runs, device=device)
    # accuracy = evaluator.exec(model=model, dataloader=test_dataloader, k=k_shot)
    # print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
