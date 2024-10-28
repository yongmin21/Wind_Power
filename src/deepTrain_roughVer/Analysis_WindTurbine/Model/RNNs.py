import torch
from torch import nn as nn
from src.deepTrain_roughVer.Analysis_WindTurbine.Model.SeriesDecomp import series_decomp_multi

# Modules
class RNNModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1,bidirectional=False):
        super(RNNModule, self).__init__()

        # 기본 RNN 레이어 추가
        self.rnn = nn.RNN(input_dim, hidden_dim, bidirectional=bidirectional, dropout=rec_dropout, batch_first=True, num_layers=num_layers)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent

class LSTMModule(nn.Module):

    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(LSTMModule, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=bidirectional, batch_first=True,
                           dropout=rec_dropout, num_layers=num_layers)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        return recurrent

class GRUModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(GRUModule, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, bidirectional=bidirectional, batch_first=True,
                          dropout=rec_dropout, num_layers=num_layers)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        return recurrent

# Models
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(RNN, self).__init__()
        self.rnn_module = RNNModule(input_dim, hidden_dim, rec_dropout, num_layers, bidirectional)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim/2)
        self.fc2 = nn.Linear(hidden_dim/2, 1)

    def forward(self, x):
        recurrent = self.rnn_module(x)
        out = self.fc1(recurrent)
        out = self.fc2(out)
        return out


# just RNNs
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm_module = LSTMModule(input_dim, hidden_dim, rec_dropout, num_layers, bidirectional)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        recurrent = self.lstm_module(x)
        out = self.fc1(recurrent)
        out = self.fc2(out)
        out = torch.clamp(out, min=0)
        return out


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(GRU, self).__init__()
        self.gru_module = GRUModule(input_dim, hidden_dim, rec_dropout, num_layers, bidirectional)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        recurrent = self.gru_module(x)
        out = self.fc1(recurrent)
        out = self.fc2(out)
        out = torch.clamp(out, min=0)
        return out
    
# apply relu in fc layer & dropout in RNNs
class LSTM_relu(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(LSTM_relu, self).__init__()
        self.lstm_module = LSTMModule(input_dim, hidden_dim, rec_dropout, num_layers, bidirectional)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.fc1 = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            )
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        recurrent = self.lstm_module(x)
        out = self.fc1(recurrent)
        out = self.fc2(out)
        # out = torch.clamp(out, min=0) # capacity write later
        return out


class GRU_relu(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(GRU_relu, self).__init__()
        self.gru_module = GRUModule(input_dim, hidden_dim, rec_dropout, num_layers, bidirectional)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.fc1 = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            )
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        recurrent = self.gru_module(x)
        out = self.fc1(recurrent)
        out = self.fc2(out)
        # out = torch.clamp(out, min=0) # capacity write later
        return out

# dropout in dense too
class LSTM_relu_dropALL(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(LSTM_relu_dropALL, self).__init__()
        self.lstm_module = LSTMModule(input_dim, hidden_dim, rec_dropout, num_layers, bidirectional)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.fc1 = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            )
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        recurrent = self.lstm_module(x)
        out = self.fc1(recurrent)
        out = self.fc2(out)
        # out = torch.clamp(out, min=0) # capacity write later
        return out


class GRU_relu_dropALL(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(GRU_relu_dropALL, self).__init__()
        self.gru_module = GRUModule(input_dim, hidden_dim, rec_dropout, num_layers, bidirectional)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.fc1 = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(rec_dropout)
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(rec_dropout)
            )
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        recurrent = self.gru_module(x)
        out = self.fc1(recurrent)
        out = self.fc2(out)
        # out = torch.clamp(out, min=0) # capacity write later
        return out


# apply dropout all layer, and apply batch norm
class GRUWithBatchNorm(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(GRUWithBatchNorm, self).__init__()
        self.gru_module = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                                 bidirectional=bidirectional, dropout=rec_dropout)

        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2 if bidirectional else hidden_dim)

        if bidirectional:
            self.fc1 = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(rec_dropout))
        else:
            self.fc1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(rec_dropout))

        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        recurrent, _ = self.gru_module(x)
        if recurrent.dim() == 3:
            recurrent = recurrent[:, -1, :]  # 마지막 time step만 사용
        recurrent = self.batch_norm(recurrent)
        out = self.fc1(recurrent)
        out = self.fc2(out)
        return out


class LSTMWithBatchNorm(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(LSTMWithBatchNorm, self).__init__()
        self.lstm_module = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                                 bidirectional=bidirectional, dropout=rec_dropout)

        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2 if bidirectional else hidden_dim)

        if bidirectional:
            self.fc1 = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(rec_dropout))
        else:
            self.fc1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(rec_dropout))

        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        recurrent, _ = self.lstm_module(x)
        if recurrent.dim() == 3:
            recurrent = recurrent[:, -1, :]  # 마지막 time step만 사용
        recurrent = self.batch_norm(recurrent)
        out = self.fc1(recurrent)
        out = self.fc2(out)
        return out

class SeriesDecompLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, rec_dropout=0, num_layers=1, in_moving_mean=True, decomp_kernel=[3, 5, 7, 9], feature_wise_norm=True):
        super(SeriesDecompLSTM, self).__init__()

        self.feature_wise_norm = feature_wise_norm
        self.in_moving_mean = in_moving_mean
        self.decomp_kernel = decomp_kernel
        self.series_decomp_multi = series_decomp_multi(kernel_size=self.decomp_kernel)

        self.lstm = LSTMModule(input_dim=input_dim, hidden_dim=hidden_dim, rec_dropout=rec_dropout, num_layers=num_layers) 
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.dense_softmax = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1) 
        
    def load_state_dict(self, state_dict, strict=True):
        self.series_decomp_multi.load_state_dict(state_dict["series_decomp_multi"])
        self.lstm.load_state_dict(state_dict["lstm"])
        self.dense.load_state_dict(state_dict["dense"])
        self.dense_softmax.load_state_dict(state_dict["dense_softmax"])
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"series_decomp_multi": self.series_decomp_multi.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "lstm": self.lstm.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "dense": self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "dense_softmax": self.dense_softmax.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'series_decomp_multi': self.series_decomp_multi.state_dict(),
                      'lstm': self.lstm.state_dict(),
                      'dense': self.dense.state_dict(),
                      'dense_softmax': self.dense_softmax.state_dict()}
        torch.save(parameters, filename)
                        
    def forward(self, x):
        x = x.float()
        
        if self.in_moving_mean:
            moving_mean, res = self.series_decomp_multi(x)
            x = moving_mean    
    
        if self.feature_wise_norm:
            # Feature-wise Normalization
            x_min = x.min(dim=1, keepdim=True)[0]
            x_max = x.max(dim=1, keepdim=True)[0]
            x = (x - x_min) / (x_max - x_min + 1e-7)
            
        x = self.lstm(x) # [nBatch, segLeng, nHidden]

        y = self.dense(x) # [nBatch, seqLeng, output_dim]

        sof = self.dense_softmax(x)  
        sof = self.softmax(sof)
        sof = torch.clamp(sof, min=1e-7, max=1)
        pred = (y * sof).sum(1) / sof.sum(1)
        return pred