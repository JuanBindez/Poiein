import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    pairs = [line.strip().split("|") for line in lines if "|" in line]
    return pairs

def build_vocab(pairs):
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
    for question, answer in pairs:
        for word in question.split() + answer.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

class ChatDataset(Dataset):
    def __init__(self, pairs, vocab):
        self.pairs = pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        question, answer = self.pairs[idx]
        question_ids = torch.tensor(
            [self.vocab[word] for word in question.split()] + [self.vocab["<EOS>"]]
        )
        answer_ids = torch.tensor(
            [self.vocab["<SOS>"]] + [self.vocab[word] for word in answer.split()] + [self.vocab["<EOS>"]]
        )
        return question_ids, answer_ids

def collate_fn(batch):
    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded

# Modelos Encoder e Decoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        outputs, hidden = self.rnn(x)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x).unsqueeze(1)
        outputs, hidden = self.rnn(x, hidden)
        outputs = self.fc(outputs.squeeze(1))
        return outputs, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        hidden = self.encoder(src)
        outputs = []
        input_token = tgt[:, 0]  # <SOS>
        for t in range(1, tgt.size(1)):
            output, hidden = self.decoder(input_token, hidden)
            outputs.append(output)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = tgt[:, t] if teacher_force else output.argmax(1)
        return torch.stack(outputs, dim=1)


def train(model, dataloader, optimizer, criterion, epochs=10, device="cpu"):
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt)
            output_dim = output.shape[-1]
            loss = criterion(output.view(-1, output_dim), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.8f}")