import torch
import torch.nn as nn

class EdgeReconstructor(nn.Module):
    """
    边重构器 (三分类任务：Remove=0, Keep=1, Add=2)
    输出: [E, 3] Logits
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(EdgeReconstructor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2 + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # [关键] 必须是 3
            nn.Linear(hidden_dim, 3) 
        )

    def forward(self, h_l, h_p, dist):
        cat_feat = torch.cat([h_l, h_p, dist], dim=-1)
        logits = self.net(cat_feat)
        return logits