import torch
import torch.nn as nn

class MyEncoderLayer(nn.Module):
    # 생성자
    def __init__(self, input_size, attention_heads=8, feedforward_size=1024, dropout_rate=0.1):
        super(MyEncoderLayer, self).__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=attention_heads,
            dropout=dropout_rate
        )
        # Feedforward Neural Network
        self.feedforward = nn.Sequential(
            nn.Linear(input_size, feedforward_size),
            nn.ReLU(),
            nn.Linear(feedforward_size, input_size)
        )
        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(input_size)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    # 학습 진행
    def forward(self, x, mask=None):
        # Self-Attention
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)

        # Feedforward
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)

        return x

# 인코더 레이어 쌓기 (우리 코드는 1개이므로 설정x)
class MyEncoder(nn.Module):
    def __init__(self, num_layers, input_size, attention_heads=8, feedforward_size=512, dropout_rate=0.1):
        super(MyEncoder, self).__init__()

        # Stack multiple encoder layers
        self.layers = nn.ModuleList([
            MyEncoderLayer(input_size=input_size)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        # Forward pass through each layer
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_encoder_layers=1):
        super(MyModel, self).__init__()

        self.encoder = MyEncoder(num_encoder_layers, input_size)  # 인코더 등록
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids.float(), mask=torch.ones((len(input_ids), len(input_ids))))  # 인코더 학습->마스크 변경
        x = self.layer1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x