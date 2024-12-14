# Poiein

AI based on Seq2Seq

### Install Requirements

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

### Quickstart

#### create a txt file with questions and answers, you must separate the dialogs in the file as follows using a | for example: question | response

```txt
hello|hello how are you?
I'm fine and you?|I'm fine too
```

### Training

```python
from poiein import Poiein

ai = Poiein(
    model_name="test_model",
    training_data_file="data.txt",
    learning_rate=0.001,
    epochs=450,
)


ai.run_training()

```

#### After running the training you can test the model

```python

ai.run_chat(load_model_file="test_model.pth")

```