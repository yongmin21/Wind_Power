import warnings
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearnex import patch_sklearn, config_context
import os
import scipy.stats as spst
import dask
import dask.dataframe as dd
from windpowerlib.wind_speed import logarithmic_profile
from src.utils import uv_to_wsd # 윈도우에서는 앞에 src를 뺄것
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from src.utils import DataConnector
from src.metric import NMAE
from src.data_processor import *
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
import torch
from src.deepTrain_roughVer.Analysis_WindTurbine.Model.RNNs import *
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from sklearn.metrics import r2_score, mean_absolute_error
import argparse


def NMAE(y_true, y_pred):
    """NMAE 계산 함수."""
    return mean_absolute_error(y_true, y_pred) / (sum(abs(y_true)) / len(y_true)) * 100

def train_model(save_dir, model, x_train, y_train, x_test, y_test, epochs=200, batch_size=24, lr=0.001, criterion_type='SmoothL1', early_n=None, weight_decay=0.0, max_grad_norm=None):
    if not torch.cuda.is_available():
        raise RuntimeError("cuda is not available. exiting...")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model = model.cuda()

    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32).cuda()
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).cuda()
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32).cuda()
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).cuda()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    criterion = nn.MSELoss() if criterion_type == 'MSE' else nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # weight_decay for prevent overfiting

    # 기울기 클리핑
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    best_val_loss = float('inf')
    best_y_pred_list = []
    best_y_true_list = []
    best_mae = float('inf')
    best_r2 = -float('inf')
    train_losses, val_losses = [], []
    early_stop_counter = 0

    csv_data = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, y_pred_list, y_true_list = 0.0, [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                y_pred_list.extend(outputs.cpu().numpy())
                y_true_list.extend(targets.cpu().numpy())

        y_pred_list = np.array(y_pred_list).flatten()
        y_true_list = np.array(y_true_list).flatten()

        mae = mean_absolute_error(y_true_list, y_pred_list)
        nmae = NMAE(y_true_list, y_pred_list)
        r2 = r2_score(y_true_list, y_pred_list)

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(test_loader))
        csv_data.append([epoch + 1, train_loss / len(train_loader), val_loss / len(test_loader), mae, nmae, r2])

        print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(test_loader):.4f} | MAE: {mae:.4f} | NMAE: {nmae:.4f} | R2: {r2:.4f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mae, best_r2 = mae, r2
            best_y_pred_list, best_y_true_list = y_pred_list, y_true_list
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model.__class__.__name__}_best_model.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_n is not None and early_stop_counter >= early_n:
            break

    # Save training log and loss plot
    df = pd.DataFrame(csv_data, columns=["Epoch", "Train Loss", "Val Loss", "MAE", "NMAE", "R2"])
    df.to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
    plt.plot(range(len(val_losses)), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss plot (Best Val Loss at epoch {best_epoch}: {(best_val_loss/1000):.2f}k)")
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()

    plt.figure()
    plt.plot(best_y_true_list, label="ground truth")
    plt.plot(best_y_pred_list, label="Pred")
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.legend()
    plt.title("Pred and Truth Compare")
    plt.savefig(os.path.join(save_dir, "best_pred_plot.png"))
    plt.close()

    # Calculate incentive metrics
    result = pd.DataFrame({'predict_energy_kwh': best_y_pred_list, 'energy_kwh': best_y_true_list})
    result["capacity"] = result["energy_kwh"].max()  # assuming max of actual energy as capacity for example
    result["normalized_abs_error"] = abs(result.predict_energy_kwh - result.energy_kwh) / result.capacity * 100
    result['incentive'] = 0.0
    result.loc[(result.normalized_abs_error > 6) & (result.normalized_abs_error <= 8), 'incentive'] = 3.0
    result.loc[(result.normalized_abs_error <= 6), 'incentive'] = 4.0
    result.loc[result.energy_kwh < result.capacity * 0.1, 'incentive'] = 0.0

    nmae = round(result.normalized_abs_error.mean(), 2)
    total_incentive = np.floor((result.incentive * result.energy_kwh).sum())
    available_max_incentive = np.floor((4 * result.energy_kwh[result.energy_kwh >= result.capacity * 0.1])).sum()
    incentive_rate = round(total_incentive / available_max_incentive * 100, 2)

    with open(os.path.join(save_dir, "training_log.txt"), "w") as f:
        f.write(f"Final Validation Loss: {best_val_loss:.4f}\n")
        f.write(f"MAE: {best_mae:.4f}, NMAE: {nmae:.4f}, R^2: {best_r2:.4f}\n")
        f.write(f"예측정산금획득율 = {incentive_rate} %\n")
        f.write(f"예측제도정산금 = {int(total_incentive)} 원\n")
        f.write(f"devide capacity => MAE/20700: {(best_mae/20700):.4f}, MAE/79600: {(best_mae/79600):.4f}")

    print(f"Final Model saved with best validation loss: {best_val_loss:.4f}")
    print(f"MAE: {best_mae:.4f}, NMAE: {nmae} %, R^2: {best_r2:.4f}")
    print(f"예측정산금획득율 = {incentive_rate} %")
    print(f"예측제도정산금 = {int(total_incentive)} 원")

def load_dataframes(load_path, *dataframe_names):
    loaded_dataframes = {}
    for name in dataframe_names:
        file_path = os.path.join(load_path, f"{name}.pkl")
        loaded_dataframes[name] = pd.read_pickle(file_path)
        print(f"{name} loaded from {file_path}")
    return loaded_dataframes

x_dict = {
    # 모든 특성 포함 (energy_kwh 제외)
    'all_features': ['elevation', 'land_cover', 'surf_rough', 'frictional_vmax_50m', 'frictional_vmin_50m', 
                     'pressure', 'relative_humid', 'specific_humid', 'temp_air', 'storm_u_5m', 'storm_v_5m', 
                     'wind_u_10m', 'wind_v_10m', 'wind_speed', 'wind_direction', 'wind_speed_100m', 'wind_u_100m', 
                     'wind_v_100m', 'hour', 'day', 'month', 'year', 'season', 'Night', 'density', 'shear_stress', 
                     'wind_direction_cos', 'wind_direction_sin', 'period_hours', 'cluster_2', 'cluster_3', 
                     'cluster_4', 'cluster_5', 'cluster_6', 'PC1', 'PC2', 'medoid_cluster_2', 'medoid_cluster_3', 
                     'medoid_cluster_4', 'medoid_cluster_5', 'medoid_cluster_6'],

    # PCA 특성만 포함
    'pca_only': ['elevation', 'land_cover', 'surf_rough', 'frictional_vmax_50m', 'frictional_vmin_50m', 
                 'pressure', 'relative_humid', 'specific_humid', 'temp_air', 'storm_u_5m', 'storm_v_5m', 
                 'wind_u_10m', 'wind_v_10m', 'wind_speed', 'wind_direction', 'wind_speed_100m', 'wind_u_100m', 
                 'wind_v_100m', 'hour', 'day', 'month', 'year', 'season', 'Night', 'density', 'shear_stress', 
                 'wind_direction_cos', 'wind_direction_sin', 'period_hours', 'PC1', 'PC2'],

    # 클러스터 특성만 포함
    'cluster_only': ['elevation', 'land_cover', 'surf_rough', 'frictional_vmax_50m', 'frictional_vmin_50m', 
                     'pressure', 'relative_humid', 'specific_humid', 'temp_air', 'storm_u_5m', 'storm_v_5m', 
                     'wind_u_10m', 'wind_v_10m', 'wind_speed', 'wind_direction', 'wind_speed_100m', 'wind_u_100m', 
                     'wind_v_100m', 'hour', 'day', 'month', 'year', 'season', 'Night', 'density', 'shear_stress', 
                     'wind_direction_cos', 'wind_direction_sin', 'period_hours', 'cluster_2', 'cluster_3', 
                     'cluster_4', 'cluster_5', 'cluster_6'],

    # Medoid 클러스터 특성만 포함
    'medoid_only': ['elevation', 'land_cover', 'surf_rough', 'frictional_vmax_50m', 'frictional_vmin_50m', 
                    'pressure', 'relative_humid', 'specific_humid', 'temp_air', 'storm_u_5m', 'storm_v_5m', 
                    'wind_u_10m', 'wind_v_10m', 'wind_speed', 'wind_direction', 'wind_speed_100m', 'wind_u_100m', 
                    'wind_v_100m', 'hour', 'day', 'month', 'year', 'season', 'Night', 'density', 'shear_stress', 
                    'wind_direction_cos', 'wind_direction_sin', 'period_hours', 'medoid_cluster_2', 'medoid_cluster_3', 
                    'medoid_cluster_4', 'medoid_cluster_5', 'medoid_cluster_6'],

    # PCA + 클러스터 특성 포함
    'pca_cluster': ['elevation', 'land_cover', 'surf_rough', 'frictional_vmax_50m', 'frictional_vmin_50m', 
                    'pressure', 'relative_humid', 'specific_humid', 'temp_air', 'storm_u_5m', 'storm_v_5m', 
                    'wind_u_10m', 'wind_v_10m', 'wind_speed', 'wind_direction', 'wind_speed_100m', 'wind_u_100m', 
                    'wind_v_100m', 'hour', 'day', 'month', 'year', 'season', 'Night', 'density', 'shear_stress', 
                    'wind_direction_cos', 'wind_direction_sin', 'period_hours', 'cluster_2', 'cluster_3', 
                    'cluster_4', 'cluster_5', 'cluster_6', 'PC1', 'PC2'],

    # PCA + Medoid 클러스터 특성 포함
    'pca_medoid': ['elevation', 'land_cover', 'surf_rough', 'frictional_vmax_50m', 'frictional_vmin_50m', 
                   'pressure', 'relative_humid', 'specific_humid', 'temp_air', 'storm_u_5m', 'storm_v_5m', 
                   'wind_u_10m', 'wind_v_10m', 'wind_speed', 'wind_direction', 'wind_speed_100m', 'wind_u_100m', 
                   'wind_v_100m', 'hour', 'day', 'month', 'year', 'season', 'Night', 'density', 'shear_stress', 
                   'wind_direction_cos', 'wind_direction_sin', 'period_hours', 'PC1', 'PC2', 'medoid_cluster_2', 
                   'medoid_cluster_3', 'medoid_cluster_4', 'medoid_cluster_5', 'medoid_cluster_6'],

    # 클러스터 + Medoid 클러스터 특성 포함
    'cluster_medoid': ['elevation', 'land_cover', 'surf_rough', 'frictional_vmax_50m', 'frictional_vmin_50m', 
                       'pressure', 'relative_humid', 'specific_humid', 'temp_air', 'storm_u_5m', 'storm_v_5m', 
                       'wind_u_10m', 'wind_v_10m', 'wind_speed', 'wind_direction', 'wind_speed_100m', 'wind_u_100m', 
                       'wind_v_100m', 'hour', 'day', 'month', 'year', 'season', 'Night', 'density', 'shear_stress', 
                       'wind_direction_cos', 'wind_direction_sin', 'period_hours', 'cluster_2', 'cluster_3', 
                       'cluster_4', 'cluster_5', 'cluster_6', 'medoid_cluster_2', 'medoid_cluster_3', 
                       'medoid_cluster_4', 'medoid_cluster_5', 'medoid_cluster_6'],

    # PCA, 클러스터, Medoid 클러스터 모두 미포함
    'none_pca_cluster_medoid': ['elevation', 'land_cover', 'surf_rough', 'frictional_vmax_50m', 'frictional_vmin_50m', 
                                'pressure', 'relative_humid', 'specific_humid', 'temp_air', 'storm_u_5m', 'storm_v_5m', 
                                'wind_u_10m', 'wind_v_10m', 'wind_speed', 'wind_direction', 'wind_speed_100m', 
                                'wind_u_100m', 'wind_v_100m', 'hour', 'day', 'month', 'year', 'season', 'Night', 
                                'density', 'shear_stress', 'wind_direction_cos', 'wind_direction_sin', 'period_hours'],

    # 클러스터 n만 포함 (각각 정의)
    'cluster_2_only': ['elevation', 'land_cover', 'surf_rough', 'frictional_vmax_50m', 'frictional_vmin_50m', 
                       'pressure', 'relative_humid', 'specific_humid', 'temp_air', 'storm_u_5m', 'storm_v_5m', 
                       'wind_u_10m', 'wind_v_10m', 'wind_speed', 'wind_direction', 'wind_speed_100m', 'wind_u_100m', 
                       'wind_v_100m', 'hour', 'day', 'month', 'year', 'season', 'Night', 'density', 'shear_stress', 
                       'wind_direction_cos', 'wind_direction_sin', 'period_hours', 'cluster_2'],

    'cluster_3_only': ['elevation', 'land_cover', 'surf_rough', 'frictional_vmax_50m', 'frictional_vmin_50m', 
                       'pressure', 'relative_humid', 'specific_humid', 'temp_air', 'storm_u_5m', 'storm_v_5m', 
                       'wind_u_10m', 'wind_v_10m', 'wind_speed', 'wind_direction', 'wind_speed_100m', 'wind_u_100m', 
                       'wind_v_100m', 'hour', 'day', 'month', 'year', 'season', 'Night', 'density', 'shear_stress', 
                       'wind_direction_cos', 'wind_direction_sin', 'period_hours', 'cluster_3'],

    'cluster_4_only': ['elevation', 'land_cover', 'surf_rough', 'frictional_vmax_50m', 'frictional_vmin_50m', 
                       'pressure', 'relative_humid', 'specific_humid', 'temp_air', 'storm_u_5m', 'storm_v_5m', 
                       'wind_u_10m', 'wind_v_10m', 'wind_speed', 'wind_direction', 'wind_speed_100m', 'wind_u_100m', 
                       'wind_v_100m', 'hour', 'day', 'month', 'year', 'season', 'Night', 'density', 'shear_stress', 
                       'wind_direction_cos', 'wind_direction_sin', 'period_hours', 'cluster_4'],

    'cluster_5_only': ['elevation', 'land_cover', 'surf_rough', 'frictional_vmax_50m', 'frictional_vmin_50m', 
                       'pressure', 'relative_humid', 'specific_humid', 'temp_air', 'storm_u_5m', 'storm_v_5m', 
                       'wind_u_10m', 'wind_v_10m', 'wind_speed', 'wind_direction', 'wind_speed_100m', 'wind_u_100m', 
                       'wind_v_100m', 'hour', 'day', 'month', 'year', 'season', 'Night', 'density', 'shear_stress', 
                       'wind_direction_cos', 'wind_direction_sin', 'period_hours', 'cluster_5'],

    'cluster_6_only': ['elevation', 'land_cover', 'surf_rough', 'frictional_vmax_50m', 'frictional_vmin_50m', 
                       'pressure', 'relative_humid', 'specific_humid', 'temp_air', 'storm_u_5m', 'storm_v_5m', 
                       'wind_u_10m', 'wind_v_10m', 'wind_speed', 'wind_direction', 'wind_speed_100m', 'wind_u_100m', 
                       'wind_v_100m', 'hour', 'day', 'month', 'year', 'season', 'Night', 'density', 'shear_stress', 
                       'wind_direction_cos', 'wind_direction_sin', 'period_hours', 'cluster_6'],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wind Power Estimation")
    
    #common
    parser.add_argument("--region", type=str, choices=["yg", "gj"], required=True, help="Region for data: yg or gj")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results and models")

    # model params
    parser.add_argument("--model_type", type=str, choices=[
            "LSTM", "GRU", "LSTM_relu", "GRU_relu", "LSTM_relu_dropALL", "GRU_relu_dropALL", 
            "LSTMWithBatchNorm", "GRUWithBatchNorm", "SeriesDecompLSTM"
        ], required=True, help="Type of model to use")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of RNN layers")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional RNNs")

    # learning params
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--criterion_type", type=str, choices=["MSE", "SmoothL1"], default="SmoothL1", help="Loss function type")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for L2 regularization")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--early_n", type=int, default=None, help="Early stopping patience")

    # data keys argument
    parser.add_argument("--feature_keys", type=str, required=True, help="List of feature keys to use from the dataset")

    args = parser.parse_args()

    # Load Data
    if args.region == "gj":
        loaded_data = load_dataframes('src/data_gj/avg_datas', 'gj_x_train_z', 'gj_x_test_z', 'gj_y_train', 'gj_y_test')
    elif args.region == "yg":
        loaded_data = load_dataframes('src/data_yg/avg_datas', 'yg_x_train_z', 'yg_x_test_z', 'yg_y_train', 'yg_y_test')

    x_train = loaded_data[f"{args.region}_x_train_z"]
    x_test = loaded_data[f"{args.region}_x_test_z"]
    y_train = loaded_data[f"{args.region}_y_train"]
    y_test = loaded_data[f"{args.region}_y_test"]

    print(x_train.columns)
    print(args.feature_keys)
    x_train_selected = x_train[x_dict[args.feature_keys]]
    x_test_selected = x_test[x_dict[args.feature_keys]]

    # Load Model
    if args.model_type == "LSTM":
        model = LSTM(input_dim=x_train_selected.shape[1], hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidirectional=args.bidirectional)
    elif args.model_type == "GRU":
        model = GRU(input_dim=x_train_selected.shape[1], hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidirectional=args.bidirectional)
    elif args.model_type == "LSTM_relu":
        model = LSTM_relu(input_dim=x_train_selected.shape[1], hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidirectional=args.bidirectional)
    elif args.model_type == "GRU_relu":
        model = GRU_relu(input_dim=x_train_selected.shape[1], hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidirectional=args.bidirectional)
    elif args.model_type == "LSTM_relu_dropALL":
        model = LSTM_relu_dropALL(input_dim=x_train_selected.shape[1], hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidirectional=args.bidirectional)
    elif args.model_type == "GRU_relu_dropALL":
        model = GRU_relu_dropALL(input_dim=x_train_selected.shape[1], hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidirectional=args.bidirectional)
    elif args.model_type == "LSTMWithBatchNorm":
        model = LSTMWithBatchNorm(input_dim=x_train_selected.shape[1], hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidirectional=args.bidirectional)
    elif args.model_type == "GRUWithBatchNorm":
        model = GRUWithBatchNorm(input_dim=x_train_selected.shape[1], hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidirectional=args.bidirectional)
    elif args.model_type == "SeriesDecompLSTM":
        model = SeriesDecompLSTM(input_dim=x_train_selected.shape[1], hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidirectional=args.bidirectional)


    info = f"hidden{args.hidden_dim}_n_layer{args.num_layers}_f{args.feature_keys}"
    save_dir = os.path.join(args.save_dir, args.region, f"{args.model_type}_{info}")

    print(f'Save to {save_dir}.')

    config_path = os.path.join(save_dir, "config.txt")
    os.makedirs(save_dir, exist_ok=True)
    with open(config_path, "w") as f:
        f.write("Training Configuration:\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"learning_rate={args.lr}\n")
        f.write(f"criterion_type={args.criterion_type}\n")
        f.write(f"early_n={args.early_n}\n")
        f.write(f"weight_decay={args.weight_decay}\n")
        f.write(f"max_grad_norm={args.max_grad_norm}\n")

    train_model(
        save_dir=save_dir,
        model=model,
        x_train=x_train_selected,
        y_train=y_train,
        x_test=x_test_selected,
        y_test=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        criterion_type=args.criterion_type,
        early_n=args.early_n,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm
    )
