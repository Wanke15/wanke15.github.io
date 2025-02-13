import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def init(self, input_dim, output_dim):
        super().__init()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class TopKRouter(nn.Module):
    def init(self, input_dim, num_experts, top_k):
        super().__init()
        self.projection = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, inputs):
        scores = self.projection(inputs)
        top_k_score, top_k_indices = torch.topk(scores, self.top_k, dim=1)
        probabilities = F.softmax(top_k_score, dim=1)
        return probabilities, top_k_indices

class MixtureOfExperts(nn.Module):
    def init(self, input_dim, output_dim, num_experts, expert_dim, top_k):
        super().__init()
        self.router = TopKRouter(input_dim, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(expert_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # 获取路由概率和Top-K专家索引
        probabilities, export_indices = self.router(inputs)
        # 将数据广播到所有专家
        inputs = inputs.unsqueeze(1).expand(-1, len(self.experts), -1)
        # 选择对应的专家处理输入数据，即只有expert_indices位置上才给inputs数据，其他位置为0
        expert_inputs = torch.zeros_like(inputs).scatter(1, export_indices, inputs)
        expert_results = [expert(expert_inputs[:, i, :]) for i, expert in enumerate(self.experts)]
        # 合并专家输出
        combined_results = torch.stack(expert_results, dim=1)
        # 使用门控网络调整权重
        gate_weights = self.gate(combined_results.mean(dim=0))
        # 加权求和得到最终输出
        final_output = torch.sum(combined_results * gate_weights.unsqueeze(0), dim=1)
        return final_output, probabilities
