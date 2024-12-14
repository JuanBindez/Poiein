# Poiein

AI based on Seq2Seq

### install Requirements

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

### Quickstart

```python
from poiein import Poiein

ai = Poiein(
    learning_rate=0.001,
    epochs=250,
)

ai.run_training()
ai.run_chat()

```