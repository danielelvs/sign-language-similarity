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
from representations.base import BaseRepresentation
from utils.args import Args
from utils.logs import Logs
from utils.loss import ContrastiveLoss



def main():
    # load arguments
    args = Args()

    seed = args.random_seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim = args.feat_dim
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    # shuffle_data = args.shuffle_data
    num_workers = args.num_workers
    num_runs = args.num_runs
    feat_dim = args.feat_dim
    k_shot = args.k_shot
    model_name = args.model_name
    dataset_name = args.dataset_name
    representation = args.representation


    # load image representations
    image_type = BaseRepresentation(representation).get_type()
    if image_type == None:
        Logs(level=logging.ERROR, msg=f"Invalid {representation} representation type.")


    # load model
    model = BaseRepresentation(model_name).get_model()

    transform = transforms.Compose([
        model.image_size,
        transforms.ToTensor(),
    ])


    # load dataset
    data = Data(dataset_name, image_type, transform)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model(feat_dim=feat_dim).to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ############################# DATASETS #############################

    # treinamento
    train_dataset = TrainingData(X=X_train, y=y_train, transform=transform)
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size)


    # avaliação
    eval_dataset = EvaluateData(X=X_val, y=y_val, transform=transform)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)

    # teste
    test_dataset = EvaluateData(X=X_test, y=y_test, transform=transform)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)


    ############################# EXECUCOES #############################

    # treinamento
    train_evaluator = Evaluate(labels=y_val, k_shot=k_shot, num_runs=num_runs, device=device)
    best_epoch, best_accuracy, loss_history, accuracy_history = Training(
    model=model, train_dataloader=train_dataloader, val_dataloader=eval_dataloader, num_epochs=epochs, criterion=criterion, optimizer=optimizer, device=device, evaluator=train_evaluator, k_shot=k_shot)
    print(f"Best Epoch {best_epoch} Best Accuracy {best_accuracy:3f}")
    Training.chart(epochs, loss_history, accuracy_history)

    # avaliação
    model.load_state_dict(torch.load("checkpoints/best-model.pth", weights_only=True))
    evaluator = Evaluate(labels=y_test, k_shot=k_shot, num_runs=num_runs, device=device)
    accuracy = evaluator.exec(model=model, dataloader=test_dataloader, k=k_shot)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
