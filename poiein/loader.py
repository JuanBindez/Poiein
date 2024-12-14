import torch
import torch.nn as nn
from poiein import Encoder, Decoder, Seq2Seq

def load_model(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
    vocab = checkpoint["vocab"]
    vocab_size = len(vocab)
    embed_size = 64
    hidden_size = 128

    encoder = Encoder(vocab_size, embed_size, hidden_size)
    decoder = Decoder(vocab_size, embed_size, hidden_size)
    model = Seq2Seq(encoder, decoder)
    model.load_state_dict(checkpoint["model_state"])
    return model, vocab