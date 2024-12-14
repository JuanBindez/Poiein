from poiein import Poiein

ai = Poiein(
    model_name="test_model",
    training_data_file="data.txt",
    learning_rate=0.001,
    epochs=450,
)


ai.run_training()

ai.run_chat(load_model_file="test_model.pth")