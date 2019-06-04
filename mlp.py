from mxnet.gluon.nn import HybridBlock, HybridSequential, Dense, Dropout
from mxnet import nd, gluon, gpu
from dataclasses import dataclass
from network import Network, Config
import numpy as np

class MLP(HybridBlock):
    def __init__(self, training=False, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.layer1 = HybridSequential()
        self.layer1.add(Dense(1024, in_units=25*25, activation="relu"),
                        Dropout(0.1 if training else 0.0))
        self.layer2 = HybridSequential()
        self.layer2.add(Dense(512, activation="relu"),
                        Dropout(0.1 if training else 0.0))
        self.layer3 = Dense(256, activation="relu")
        self.output = Dense(1)
        self.hybridize()

    def hybrid_forward(self, F, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)

CONFIG = Config(
    context = gpu(0),
    loss = gluon.loss.L2Loss(),
    optimizer = "adam",
    learning_rate = 0.01,
    epochs = 100,
    batch_size = 32,
    model_base = "./model/mlp",
    dataset_size = 5000,
    init_steps = 5
)

def transform(board):
    onehot = np.zeros((25, 25))
    onehot[np.arange(25), board.state] = 1
    return onehot.flatten() # 625 one-hot vector

def network(training=True):
    return Network(MLP(training), CONFIG, transform)
