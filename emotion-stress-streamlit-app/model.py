import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2_LSTM_MultiTask(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.shared_fc = nn.Linear(512, 256)
        self.emotion_head = nn.Linear(256, num_emotions)
        self.stress_head = nn.Linear(256, 1)

    def forward(self, input_values):
        outputs = self.wav2vec(input_values)
        x = outputs.last_hidden_state

        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)

        shared = torch.relu(self.shared_fc(pooled))
        return self.emotion_head(shared), self.stress_head(shared)
