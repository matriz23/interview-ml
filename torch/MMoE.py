import torch
import torch.nn as nn
import torch.optim as optim


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output


class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        return output


class MMoE(nn.Module):
    def __init__(
        self,
        input_size,
        num_experts,
        expert_output_size,
        expert_hidden_size,
        tower_hidden_size,
        tasks,
    ):
        super(MMoE, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.expert_output_size = expert_output_size
        self.expert_hidden_size = expert_hidden_size
        self.tower_hidden_size = tower_hidden_size
        self.tasks = tasks

        self.softmax = nn.Softmax(dim=1)

        self.experts = nn.ModuleList(
            [
                Expert(self.input_size, self.expert_output_size, expert_hidden_size)
                for _ in range(self.num_experts)
            ]
        )
        self.towers = nn.ModuleList(
            [Tower(expert_output_size, 1, tower_hidden_size) for _ in range(self.tasks)]
        )
        self.w_gates = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(self.input_size, self.num_experts), requires_grad=True
                )
                for _ in range(self.tasks)
            ]
        )

    def forward(self, x):
        outputs = [
            expert(x) for expert in self.experts
        ]  # (bs, expert_output_size) * num_experts
        output_tensor = torch.stack(outputs)  # (num_experts, bs, expert_output_size)
        gates = [
            self.softmax(torch.matmul(x, w_gate)) for w_gate in self.w_gates
        ]  # (bs, num_experts) * tasks

        tower_inputs = [
            gate.t().unsqueeze(2).expand(-1, -1, self.expert_output_size)
            * output_tensor
            for gate in gates
        ]  # (num_experts, bs, expert_output_size) * tasks
        tower_inputs = [
            torch.sum(tower_input, dim=0) for tower_input in tower_inputs
        ]  # (bs, expert_output_size) * tasks

        output = [
            tower(tower_input) for tower, tower_input in zip(self.towers, tower_inputs)
        ]

        return output
