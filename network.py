from AStar import Board
from mxnet import nd, gluon, init
from mxnet.gluon import nn, Trainer
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class Config:
    # Configuration of training
    loss       : gluon.loss.Loss
    optimizer  : str
    epochs     : int
    batch_size : int
    params_file   : str
    # Configuration of trainer
    learning_rate : float
    momentum      : Optional[float]
    weight_decay  : Optional[float]

class Network:
    def __init__(self, net: nn.Block, cfg: Config, predict):
        self.net = net
        if os.path.exists(cfg.params_file):
            self.net.load_parameters(cfg.params_file)
        else:
            print("Params file not found. Initializing...")
            self.net.initialize(init=init.Xavier())
            self.net.save_parameters(cfg.params_file)
        self.net_predict = predict
        self.loss = cfg.loss
        self.trainer = Trainer(self.net.params(), cfg.optimizer, {
            "learning_rate": cfg.learning_rate,
            "momentum": cfg.momentum,
            "wd": cfg.weight_decay
        })

    def predict(self, cur: Board, goal: Board) -> int:
        return self.net_predict(self.net, cur.state, goal.state)
