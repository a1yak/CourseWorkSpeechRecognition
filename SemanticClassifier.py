import torch.nn as nn
import torch

class SemanticClassifier(nn.Module):
    """Neural Network for semantic text analysis"""

    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=128, num_classes=5):
        super(SemanticClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)

        x = self.fc1(attended)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
