from loader import load_model
from poiein import chat
from poiein.train import *

class Poiein:
    def __init__(self, model_name: str = "Poieien_model",
                training_data_file: str = "data.txt",
                learning_rate: float = 0.001, 
                epochs: int = 100):

        self.model = model_name + ".pth"
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.training_data_file = training_data_file

    def run_training(self):
        filepath = self.training_data_file
        pairs = load_data(filepath)

        vocab = build_vocab(pairs)

        dataset = ChatDataset(pairs, vocab)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

        vocab_size = len(vocab)
        embed_size = 64
        hidden_size = 128
        learning_rate = self.learning_rate
        epochs = self.epochs

        encoder = Encoder(vocab_size, embed_size, hidden_size)
        decoder = Decoder(vocab_size, embed_size, hidden_size)
        model = Seq2Seq(encoder, decoder)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train(model, dataloader, optimizer, criterion, epochs, device="cuda" if torch.cuda.is_available() else "cpu")

        torch.save({"model_state": model.state_dict(), "vocab": vocab}, self.model)
        print(f"Model saved as {self.model}")

    
    def run_chat(self):
        model, vocab = load_model(self.model_file)
        print("Type 'leave' to end the chat.")
        chat(model, vocab)