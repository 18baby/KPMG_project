import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Batch normalization layer
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # 모델의 forward 계산 로직 추가
        x = self.layer1(input_ids.float())
        x = self.batch_norm(x)  # Apply batch normalization
        x = self.relu(x)
        x = self.layer2(x)

        return x