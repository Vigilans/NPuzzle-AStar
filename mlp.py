from mxnet.gluon.nn import Block, Dense, Dropout, Sequential
from mxnet import nd, gluon, gpu
from dataclasses import dataclass
from network import Network, Config
import numpy as np

class MLP(Block):
    def __init__(self, training=False, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.layer1 = Sequential()
        self.layer1.add(Dense(1024, in_units=25*25, activation="relu"),
                        Dropout(0.1 if training else 0.0))
        self.layer2 = Sequential()
        self.layer2.add(Dense(512, activation="relu"),
                        Dropout(0.1 if training else 0.0))
        self.layer3 = Dense(256, activation="relu")
        self.output = Dense(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)

CONFIG = Config(
    context = gpu(0),
    loss = gluon.loss.L2Loss(),
    optimizer = "adam",
    learning_rate = 0.01,
    epochs = 200,
    batch_size = 32,
    params_file = "./model/mlp.params"
)

def transform(board):
    onehot = np.zeros((25, 25))
    onehot[np.arange(25), board.state] = 1
    return onehot.flatten() # 625 one-hot vector

def network(training=True):
    return Network(MLP(training), CONFIG, transform)
