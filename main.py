from poiein import Poiein

ai = Poiein(
    learning_rate=0.001,
    epochs=450,
)


ai.run_training()
ai.run_chat()