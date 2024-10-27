import os
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from SCADA_data import T1Dataset
from Model.RNNs import LSTMModule, RNNModule, GRUModule
from utility import weights_init

# configs
batch_size = 32
learning_rate = 0.001
num_epochs = 100

hidden_dim = 128
dropout = 0.2
num_layers = 2

model_types = [
                "RNN",
                "LSTM",
                "GRU",
                "Bidirectional-RNN",
                "Bidirectional-LSTM",
                "Bidirectional-GRU",
               ]

def train(train_loader, val_loader, model, save_dir, show_fig = False):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # train
        for data in train_loader:
            inputs, targets = data
            inputs = torch.tensor(inputs).cuda() 
            targets = torch.tensor(targets).cuda() 
            
            optimizer.zero_grad()
    
            outputs = model(inputs)
    
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                inputs, targets = data
                inputs = torch.tensor(inputs).cuda() 
                targets = torch.tensor(targets).cuda() 
                
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))

                val_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        epoch_val_loss = val_loss / len(val_loader)        
        val_loss_history.append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")
        
        torch.save(model.state_dict(), os.path.join(save_dir,f"model_epoch_{epoch+1}"))
        
        if epoch == 0 or epoch_val_loss < min(val_loss_history[:-1]):
            torch.save(model.state_dict(), os.path.join(save_dir,"best_model"))

    np.save(os.path.join(save_dir,'loss_history.npy'), np.array(loss_history))
    np.save(os.path.join(save_dir,'val_history.npy'), np.array(val_loss_history))
    
    plt.plot(loss_history, label="Train Loss")
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(save_dir,'training_loss.png'))
    
    if show_fig:
        plt.show()
    
    return os.path.join(save_dir,"best_model")

def test(test_loader, model_path, save_dir, show_fig = False):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    criterion = torch.nn.MSELoss()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs = torch.tensor(inputs).cuda()
            targets = torch.tensor(targets).cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            
            test_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    # test 평균 mse
    test_loss = test_loss / len(test_loader)
    rmse = np.sqrt(test_loss)
    print(f"Test RMSE: {rmse:.4f}")
    
    # 예측이랑 실제 값(1d vector)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    plt.figure(figsize=(10,6))
    plt.plot(all_targets, label="Actual", color="blue")
    plt.plot(all_preds, label="Predicted", color="red")
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.title('Predicted result')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'test_results.png'))
    
    if show_fig:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wind Power Estimation")

    # common
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results and models")
    parser.add_argument("--model_type", type=str, choices=["GRU", "LSTM", "RNN"], required=True, help="Type of model to use")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--criterion_type", type=str, choices=["MSE", "SmoothL1"], default="MSE", help="Loss function type")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for L2 regularization")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--early_n", type=int, default=None, help="Early stopping patience")

    # data key
    parser.add_argument("--feature_keys", nargs='+', required=True, help="List of feature keys to use from the dataset")

    args = parser.parse_args()
    
    # 배치사이즈, 에퐄, 레이어 등은 전역 변수에 둠.
    # 추후 yaml로
    
    # 모델 종류도 바꿔야 하는데, 아직은 parse arg로 받음.
    
    flags = parser.parse_args()
    
    # 바로 정해줄 친구들
    save_dir = flags.save_dir
    
    # save dir (추후 지정 가능도록 할 예정)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    train_dataset = T1Dataset("archive/train_data.csv")
    val_dataset = T1Dataset("archive/val_data.csv")
    test_dataset = T1Dataset("archive/test_data.csv")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = train_dataset.data.shape[1]
    output_dim = 1
    
    if flags.model_type == "RNN":
        model = RNNModule(input_dim, hidden_dim, rec_dropout=dropout, num_layers=num_layers, bidirectional=False)

    elif flags.model_type == "LSTM":
        model = LSTMModule(input_dim, hidden_dim, rec_dropout=dropout, num_layers=num_layers, bidirectional=False)

    elif flags.model_type == "GRU":
        model = GRUModule(input_dim, hidden_dim, rec_dropout=dropout, num_layers=num_layers, bidirectional=False)

    elif flags.model_type == "Bidirectional-RNN":
        model = RNNModule(input_dim, hidden_dim, rec_dropout=dropout, num_layers=num_layers, bidirectional=True)

    elif flags.model_type == "Bidirectional-LSTM":
        model = LSTMModule(input_dim, hidden_dim, rec_dropout=dropout, num_layers=num_layers, bidirectional=True)

    elif flags.model_type == "Bidirectional-GRU":
        model = GRUModule(input_dim, hidden_dim, rec_dropout=dropout, num_layers=num_layers, bidirectional=True)

    else:
        raise ValueError(f"Unknown model type: {flags.model_type}")
    model.cuda()
    model.apply(weights_init)
    
    model_path = None
    
    # Train Mode
    if flags.mode == "train":
        model_path = train(train_loader, val_loader, model, save_dir)
    
    # Test Mode
    
    # train아니면 path가 None임.
    if model_path == None:
        model_path = flags.load_path
        
    test(test_loader, model_path, save_dir)