import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class PhysicsConsistencyLoss(nn.Module):
    """
    [独特损失函数] 物理一致性损失
    逻辑：边重构网络的判断 (Edge Probability) 应该与物理模拟网络计算出的能量 (Interaction Energy) 保持一致。
    
    公式灵感：
    Label_pseudo = Sigmoid(-Energy)  (能量越低/越负，应该是边的概率越大)
    Loss = BCE(Prob, Label_pseudo) + Sparsity_Penalty
    """
    def __init__(self, sparsity_weight=1e-3):
        super(PhysicsConsistencyLoss, self).__init__()
        self.sparsity_weight = sparsity_weight
        self.bce = nn.BCELoss()

    def forward(self, edge_probs, edge_energies):
        """
        edge_probs: [E, 1] 边重构概率 (0~1)
        edge_energies: [E, 1] 物理能量总和 (VDW + Elec + HBond)
        """
        # 1. 构造伪标签 (Pseudo Label)
        # 能量越小(负值)，结合越紧密，概率应越接近 1
        # 我们使用 Sigmoid(-E) 将能量映射到 0~1
        # 注意：为了数值稳定性，可以对能量进行一定的缩放
        with torch.no_grad():
            target_probs = torch.sigmoid(-edge_energies)
        
        # 2. 一致性损失 (Cross Entropy)
        # 迫使重构器的判断去拟合物理规律
        consistency_loss = self.bce(edge_probs, target_probs)
        
        # 3. 稀疏性惩罚 (L1 Regularization)
        # 我们希望保留的边尽可能少，只保留关键边
        sparsity_loss = torch.mean(torch.abs(edge_probs))
        
        return consistency_loss + self.sparsity_weight * sparsity_loss


class PCGradOptimizer:
    """
    [Pareto 优化] Projected Conflicting Gradients (PCGrad)
    
    基于 Pareto 理论的多任务优化器。
    原理：
    1. 分别计算 Task 1 (Affinity) 和 Task 2 (Edge Reconstruction) 的梯度。
    2. 计算两个梯度的余弦相似度。
    3. 如果梯度冲突 (夹角 > 90度, dot product < 0)，则将其中一个梯度投影到另一个梯度的法向量上。
    4. 从而找到一个由于两个任务的“Pareto 下降方向”。
    """
    def __init__(self, optimizer):
        self._optim = optimizer

    def zero_grad(self):
        self._optim.zero_grad()

    def step(self):
        self._optim.step()

    def pc_backward(self, objectives):
        """
        objectives: list of Tensor, [loss_affinity, loss_edge]
        """
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads)
        self._unflatten_grad(pc_grad, shapes, has_grads)

    def _project_conflicting(self, grads):
        # grads: list of flattened gradient vectors for each task
        # pc_grad: projected gradients
        pc_grad = copy.deepcopy(grads)
        
        # 随机打乱任务顺序以避免偏差
        random.shuffle(pc_grad)
        
        # 对每个任务的梯度进行投影
        for g_i in pc_grad:
            for g_j in grads:
                # 计算点积 (g_i . g_j)
                g_i_g_j = torch.dot(g_i, g_j)
                
                # 如果冲突 (点积 < 0)
                if g_i_g_j < 0:
                    # g_i = g_i - (g_i . g_j) / ||g_j||^2 * g_j
                    # 也就是减去 g_i 在 g_j 方向上的分量
                    norm_g_j = g_j.norm()**2
                    if norm_g_j > 1e-8: # 防止除零
                         g_i -= (g_i_g_j / norm_g_j) * g_j
        
        # 将所有修正后的梯度相加 (这里我们简单相加，也可以求平均)
        final_grad = torch.zeros_like(grads[0])
        for g in pc_grad:
            final_grad += g
            
        return final_grad

    def _pack_grad(self, objectives):
        """计算每个 Loss 的梯度并展平为一维向量"""
        grads = []
        shapes = []
        has_grads = []
        
        for loss in objectives:
            # 保留计算图，因为我们要多次 backward
            self._optim.zero_grad()
            loss.backward(retain_graph=True)
            
            grad_list = []
            shape_list = []
            has_grad_list = []
            
            for param in self._optim.param_groups[0]['params']:
                if param.grad is not None:
                    grad_list.append(param.grad.view(-1))
                    shape_list.append(param.shape)
                    has_grad_list.append(True)
                else:
                    shape_list.append(param.shape)
                    has_grad_list.append(False)
            
            if len(grad_list) > 0:
                grads.append(torch.cat(grad_list))
            else:
                # 极端情况：Loss 与参数无关联
                grads.append(torch.zeros(0))
                
            shapes.append(shape_list)
            has_grads.append(has_grad_list)
            
        return grads, shapes[0], has_grads[0]

    def _unflatten_grad(self, pc_grad, shapes, has_grads):
        """将投影后的梯度赋值回 param.grad"""
        idx = 0
        for i, param in enumerate(self._optim.param_groups[0]['params']):
            if has_grads[i]:
                numel = param.numel()
                g = pc_grad[idx : idx + numel]
                param.grad = g.view(shapes[i])
                idx += numel
            else:
                param.grad = None
