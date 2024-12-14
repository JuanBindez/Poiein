from poiein.loader import *
from poiein.train import *
from poiein.prompt import chat


class Poiein:
    def __init__(self,
                model_name: str = "Poiein_model",
                training_data_file: str = "data.txt",
                learning_rate: float = 0.001, 
                epochs: int = 100
        ):

        self.model_name = model_name + ".pth"
        self.training_data_file = training_data_file
        self.learning_rate = learning_rate
        self.epochs = epochs

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

        torch.save({"model_state": model.state_dict(), "vocab": vocab}, self.model_name)
        print(f"Model saved as {self.model_name}")

    
    def run_chat(self, load_model_file: str = "Poiein_model.pth"):
        model, vocab = load_model(load_model_file)
        print("Type 'leave' to end the chat.")
        chat(model, vocab)