#network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CFRNet(nn.Module):
    """
    Deep CFRのためのニューラルネットワークモデル.
    状態ベクトルを入力とし、各行動のカウンターファクチュアル後悔値を出力する.
    """
    def __init__(self, state_dim, action_dim):
        """
        Args:
            state_dim (int): 入力される状態ベクトルの次元数
            action_dim (int): 行動の数（=出力次元数）
        """
        super(CFRNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        """
        順伝播の定義
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        regret_values = self.fc3(x)
        return regret_values