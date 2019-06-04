from mlp import *
from train import *

if __name__ == "__main__":
    mlp = network()
    trainer = Trainer(mlp)
    trainer.start()
