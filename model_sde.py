import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa import STFT
from pytorch_lightning import LightningModule
import pandas as pd


class SDENet_RIR(nn.Module):
    def __init__(self,
                 num_freq_bins,
                 seq_len,
                 lstm_hidden_size,
                 lstm_num_layers,
                 output_size=1
                 ):
        super(SDENet_RIR, self).__init__()
        # spectrogram representation of RIR
        self.n_fft = 512
        self.hop_length = 256

        self.input_size = num_freq_bins
        self.seq_len = seq_len
        self.output_size = output_size

        # LSTM variables
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        self.lstm = nn.LSTM(self.input_size, 
                            self.lstm_hidden_size, 
                            num_layers=self.lstm_num_layers,
                            batch_first=True)
        self.fcn = nn.Linear(seq_len, output_size)
    
    def forward(self, x):
        """
        Input:
            x : RIR
        
        Output:
            out : distance estimate
        """
        out = self.lstm(x)
        out = self.fcn(out)
        return out

class SDENet_Source(nn.Module):
    def __init__(self,
                 ):
        super(SDENet_Source, self).__init__()

        # spectrogram representation of

    
    def forward(self, x):
        """
        Input:
            x : speech
        
        Output:
            d : distance estimate
        """

if __name__ == "__main__": 

    x = torch.rand((1,16000))

    X = torch.stft(x, 256, 128, return_complex=True)

    model = SDENet_RIR(129, 126, 3, 1)
    y = model(X)
    print(y.shape)