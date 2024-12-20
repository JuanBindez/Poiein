from poiein import Poiein
import os

ai = Poiein(
    model_name="test_model",
    training_data_file="data.txt",
    learning_rate=0.001,
    epochs=150,
)

ai.run_training()
os.system("clear")


ai.run_chat(load_model_file="test_model.pth")