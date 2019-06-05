# from mlp import *
from cnn import *
from train import *

if __name__ == "__main__":
    trainer = Trainer(network)
    trainer.start()
