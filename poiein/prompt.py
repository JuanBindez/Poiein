import torch

def chat(model, vocab):
    reverse_vocab = {v: k for k, v in vocab.items()}
    model.eval()
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"sair", "exit"}:
            break
        input_tensor = torch.tensor([[vocab.get(word, vocab["<PAD>"]) for word in user_input.split()] + [vocab["<EOS>"]]], dtype=torch.long)
        with torch.no_grad():
            hidden = model.encoder(input_tensor)
            input_token = torch.tensor([vocab["<SOS>"]])
            response = []
            for _ in range(40):  # Limite de 20 palavras na resposta
                output, hidden = model.decoder(input_token, hidden)
                next_word = output.argmax(1).item()
                if next_word == vocab["<EOS>"]:
                    break
                response.append(reverse_vocab[next_word])
                input_token = torch.tensor([next_word])
        print("Chatbot:", " ".join(response))