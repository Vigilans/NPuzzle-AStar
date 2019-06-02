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
        self.cfg = cfg
        self.net_predict = predict
        self.loss = cfg.loss
        self.trainer = None
        self.load_trainer()
        if os.path.exists(cfg.params_file):
            self.net.load_parameters(cfg.params_file)
        else:
            print("Params file not found. Initializing...")
            self.net.initialize(init=init.Xavier())
            self.net.save_parameters(cfg.params_file)

    def load_trainer(self):
        opt = self.cfg.optimizer
        opt_dict = {
            "learning_rate": self.cfg.learning_rate,
            "momentum": self.cfg.momentum,
            "wd": self.cfg.weight_decay
        }
        if self.trainer is None:
            self.trainer = Trainer(self.net.params(), opt, opt_dict)
        else:
            self.trainer._init_optimizer(opt, opt_dict)

    def predict(self, cur: Board, goal: Board) -> int:
        return self.net_predict(self.net, cur.state, goal.state)
