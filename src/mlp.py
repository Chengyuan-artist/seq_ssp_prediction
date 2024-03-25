import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dataset import ProteinDataset

# import torch.nn.functional as F


# input: (BACH_SIZE, LEN, 20)
# Linear: (20, n) -> ReLU -> (n, 3)
class MLP(nn.Module):
    def __init__(
        self,
        split_len=20,
        hidden=64,
        epochs=100,
        batch_size=32,
        learning_rate=1e-4,
    ):
        super().__init__()
        self.split_len = split_len
        dim_in = split_len * 20
        feat_dim = split_len * 3
        self.fcs = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim),
        )
        self.softmax = nn.Softmax(dim=-1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def forward(self, x):
        h = self.fcs(x)
        h = h.view(-1, self.split_len, 3)
        x = self.softmax(h)
        x = x.view(-1, self.split_len * 3)
        return x

    def save_model(self, save_file):
        print("==> Saving...")
        state = {
            "model": self.state_dict(),
            "epoch": self.epochs,
        }
        torch.save(state, save_file)
        del state

    def load_data(self, test_data, valid_data):

        self.train_loader = DataLoader(
            test_data, batch_size=self.batch_size, shuffle=True
        )
        self.valid_loader = DataLoader(
            valid_data, batch_size=self.batch_size, shuffle=True
        )

        
    def train_model(self, verbose=False):
        self = self.to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.loss_df = pd.DataFrame(columns=["epoch", "loss"])

        for epoch in tqdm(range(self.epochs), desc="epoch", leave=False):
            # if verbose:
            #     print(f"Starting epoch {epoch+1}")
            self.train()
            prog_iter = tqdm(self.train_loader, desc="Training", leave=False)
            current_loss = 0.0

            for i, data in enumerate(prog_iter):
                inputs, targets = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.loss_function(outputs, targets)
                loss.backward()
                self.optimizer.step()
                current_loss += loss.item()

            self.loss_df = self.loss_df._append(
                {"epoch": epoch, "loss": current_loss / len(self.train_loader)},
                ignore_index=True,
            )
        
        # plt.plot(self.loss_df["epoch"], self.loss_df["loss"])

    def save_res(self, save_file):
        self.loss_df.to_csv(f"{save_file}_loss.csv", index=False)
        self.save_model(save_file)
        # plt.plot(self.loss_df["epoch"], self.loss_df["loss"])

    def load_model(self, save_file):
        saved_params = torch.load(save_file)
        self.load_state_dict(saved_params["model"])
        self = self.to(self.device)
        epoch = saved_params["epoch"]
        print(f"Loaded model: {save_file}")

    def valid(self, verbose=False):
        self.eval()    
        with torch.no_grad():
            q3s = []
            for data in self.valid_loader:
                inputs, targets = data[0], data[1]
                for seq, ssp in zip(inputs, targets):
                    
                    split_seq = ProteinDataset.split_seq(seq, self.split_len)
                    seq_tensors = [ProteinDataset.seq_to_onehot(seq, reqular_len=self.split_len).view(-1) for seq in split_seq]
                    input_tensor = torch.stack(seq_tensors)
                    input_tensor = input_tensor.to(self.device)
                    # print(input_tensor.shape)
                    output_tensor = self(input_tensor)
                    # print(output_tensor.shape)
                    
                    one_hot = ProteinDataset.prob_to_onehot(output_tensor.view(-1, 3))
                    # print(one_hot.shape)
                    ssp_predicted = ProteinDataset.onehot_to_sequence(one_hot, is_ssp=True)
                    if verbose:
                        print(f"True_len: {len(ssp)}")
                        print(f"Pred_len: {len(ssp_predicted)}")
                        print(f"True: {ssp}")
                        print(f"Pred: {ssp_predicted}")
                        
                    q3 = self.q3(ssp, ssp_predicted)
                    q3s.append(q3)
                    if verbose:
                        print(f"Q3: {q3}")
                        
        return q3s
                    
    def q3(self, ssp_true, ssp_predicted):
        q3 = sum([1 for i in range(len(ssp_true)) if ssp_true[i] == ssp_predicted[i]]) / len(ssp_true)
        return q3
    
# print(ProteinDataset.onehot_to_sequence(torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), is_ssp=True))