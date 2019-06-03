from mlp import *
from train import *

mlp = network()
trainer = Trainer(mlp)
trainer.start()
