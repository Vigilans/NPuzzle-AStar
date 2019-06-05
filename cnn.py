from mxnet.gluon.nn import HybridBlock, HybridSequential, Dense, Dropout, Conv2D, MaxPool2D
from mxnet import nd, gluon, gpu
from dataclasses import dataclass
from network import Network, Config
from AStar import Board
import numpy as np

class CNN(HybridBlock):
    def __init__(self, training=False, **kwargs):
        super(CNN, self).__init__(**kwargs)
        self.cnn = HybridSequential()
        self.cnn.add( # We don't need pooling, since local information matters
            Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'), # Sees 3*3
            Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'), # Sees 5*5
            Dense(units=1024, activation='relu'),
            Dropout(0.2 if training else 0.0),
            Dense(units=512, activation='relu'),
            Dropout(0.2 if training else 0.0),
            Dense(units=256, activation='relu'),
            Dense(1))
        self.cnn.hybridize()

    def hybrid_forward(self, F, x):
        return self.cnn(x)

CONFIG = Config(
    context = gpu(0),
    loss = gluon.loss.L2Loss(),
    optimizer = "adam",
    learning_rate = 0.005,
    epochs = 10,
    batch_size = 32,
    model_base = "./model/cnn",
    dataset_size = 10000,
    init_steps = 10
)

def transform(board):
    state = board.state.reshape((5, 5))
    goal  = Board.ordered().state.reshape((5, 5))
    blank = np.zeros(25)
    blank[board.blank_tile] = 1
    return np.stack([state, goal, blank.reshape(5, 5)])

def network(training=True):
    return Network(CNN(training), CONFIG, transform)
