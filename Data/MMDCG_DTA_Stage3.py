import torch
import torch.nn as nn
from MMDCG_DTA_Stage2 import MMDCGDTAModel_Stage2

class MMDCGDTAModel_Stage3(MMDCGDTAModel_Stage2):
    """
    Stage 3 模型：最终亲和力微调阶段。
    继承关系：Stage 3 -> Stage 2 -> Stage 1 (包含所有物理注入逻辑)。
    """
    def __init__(self, config):
        super(MMDCGDTAModel_Stage3, self).__init__(config)

    def freeze_reconstructor(self):
        """
        [核心] 冻结边重构器参数。
        利用 Stage 2 训练好的物理分类能力来指导 Stage 3 的注意力机制。
        """
        print("Freezing Edge Reconstructor parameters...")
        for param in self.edge_classifier.parameters():
            param.requires_grad = False
            
    def unfreeze_reconstructor(self):
        """解冻逻辑（可选）"""
        print("Unfreezing Edge Reconstructor parameters...")
        for param in self.edge_classifier.parameters():
            param.requires_grad = True

    def forward(self, sample):
        # 调用 Stage 2 的 forward。
        # 返回: (y_pred, recon_stats, edge_logits, flat_edge_energies)
        # 即使 edge_classifier 冻结，它依然会输出 logits 并通过 Softmax 生成 edge_weights 作用于 GAT。
        return super(MMDCGDTAModel_Stage3, self).forward(sample)