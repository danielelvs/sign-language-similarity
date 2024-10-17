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
from model.resnet18 import Resnet18Model
from representations.base import BaseImageRepresentation
from utils.arguments import ArgumentsParser
from utils.logs import Logs
from utils.loss import ContrastiveLoss


def main():
    # load arguments
    args = ArgumentsParser().parse_args()
    seed = args.random_seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = args.learning_rate
    epochs = 1000 #args.epochs
    batch_size = args.batch_size
    # shuffle_data = args.shuffle_data
    num_workers = args.num_workers
    num_runs = args.num_runs
    # feat_dim = args.feat_dim
    k_shot = 1 #args.k_shot
    model_name = "Resnet18" #args.model_name
    dataset_name = 'ufop' #args.dataset_name
    image_representation = args.image_representation

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    # 1. Get Image Representation
    Logs(level=logging.INFO, message=f"1. Getting {image_representation} image representation...\n")
    image_method = BaseImageRepresentation.get_type(image_representation)
    if image_method is None:
        Logs(level=logging.ERROR, message=f"Invalid image representation type.")
        raise ValueError("Invalid image representation type.")


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Converte para 3 canais
        transforms.ToTensor(),
    ])


    # 2. Load Dataset
    Logs(level=logging.INFO, message=f"2. Loading dataset {dataset_name}...\n")
    data = Data(dataset_name=dataset_name, image_method=image_method, transform=transform)
    if data is None:
        Logs(level=logging.ERROR, message=f"Invalid dataset.")
        raise ValueError("Invalid dataset.")


    X, y, p = data.get_features()
    num_features = len(np.unique(y))
    print(num_features)

    # 3. Load Model
    # Logs(level=logging.INFO, message=f"3. Loading model {model_name}...\n")
    # base_model = BaseModel.get_by_name(model_name)(num_features)
    # model = base_model.get_model()
    # if model is None:
    #     Logs(level=logging.ERROR, message=f"Invalid model type.")
    #     raise ValueError("Invalid model type.")

    model = Resnet18Model(num_classes=num_features).to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Split dataset
    # X_train, y_train, X_val, y_val, X_test, y_test = data.split(X, y)
    # data.samples(y_train, y_val, y_test)
    # print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    # Supondo que X e y sejam arrays NumPy contendo as imagens e os rótulos
    # Verifique a forma dos dados antes da divisão
    print(f"Original X shape: {X.shape}, y shape: {y.shape}")

    # Certifique-se de que os dados estão no formato correto
    # X = (X * 255).astype(np.uint8)  # Converta para uint8 se necessário

    # Definir as transformações
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Converte para 3 canais
    ])

    # Dividir os dados em conjuntos de treinamento, validação e teste
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Verifique a forma dos dados após a divisão
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


    ############################# DATASETS #############################

    # treinamento
    train_dataset = TrainingData(X=X_train, y=y_train, transform=None)
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size)

    # avaliação
    eval_dataset = EvaluateData(X=X_val, y=y_val, transform=None)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)

    # teste
    test_dataset = EvaluateData(X=X_test, y=y_test, transform=None)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)

    ############################# EXECUCOES #############################

    # treinamento
    train_evaluator = Evaluate(labels=y_val, k_shot=k_shot, num_runs=num_runs, device=device)

    best_epoch, best_accuracy, best_loss_history, loss_history, accuracy_history = Training.exec(model=model,train_dataloader=train_dataloader,val_dataloader=eval_dataloader,num_epochs=epochs,criterion=criterion,optimizer=optimizer,device=device,evaluator=train_evaluator,k_shot=k_shot)
    print(f"Best Epoch {best_epoch} Best Accuracy {best_accuracy:3f}")
    Training.chart(epochs, loss_history, accuracy_history)

    # avaliação
    model.load_state_dict(torch.load("checkpoints/best-model.pth", weights_only=True))
    evaluator = Evaluate(labels=y_test, k_shot=k_shot, num_runs=num_runs, device=device)
    accuracy = evaluator.exec(model=model, dataloader=test_dataloader, k=k_shot)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
