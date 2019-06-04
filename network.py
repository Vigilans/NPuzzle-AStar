from AStar import Board
from mxnet import nd, gluon, init, gpu, Context
from mxnet.gluon import nn, Trainer
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class Config:
    context    : Context
    loss       : gluon.loss.Loss
    optimizer  : str
    model_base : str
    epochs     : int
    batch_size : int
    init_steps : int
    dataset_size  : int
    learning_rate : float
    momentum      : Optional[float] = None
    weight_decay  : Optional[float] = None


class Network:
    def __init__(self, net: nn.HybridBlock, cfg: Config, transform):
        self.net = net
        self.cfg = cfg
        self.transform = transform
        self.loss = cfg.loss
        self.trainer = None
        self.load_params() # Load the current best params
        self.load_trainer()

    def predict(self, cur: Board, goal: Board) -> int:
        state = nd.array([self.transform(cur)], ctx=self.cfg.context)
        return round(self.net(state)[0].asscalar())

    def load_trainer(self):
        gluon.model_zoo.vision.ResNetV2
        opt = self.cfg.optimizer
        opt_dict = { "learning_rate": self.cfg.learning_rate }
        if self.cfg.momentum is not None:
            opt_dict["momentum"] = self.cfg.momentum
        if self.cfg.weight_decay is not None:
            opt_dict["wd"] = self.cfg.weight_decay
        if self.trainer is None:
            self.trainer = Trainer(self.net.collect_params(), opt, opt_dict)
        else:
            self.trainer._init_optimizer(opt, opt_dict)

    def load_params(self, round=None):
        params_file = self.params_file(round)
        if os.path.exists(params_file):
            self.net.load_parameters(params_file, ctx=self.cfg.context)
        else:
            print("Params file not found. Initializing...")
            self.net.initialize(init=init.Xavier(), ctx=self.cfg.context)

    def save_params(self, round=None):
        self.net.save_parameters(self.params_file(round))

    def save_model(self, round):
        self.net.hybridize() # export needs hybridize and forward at least once
        self.predict(Board.ordered(), Board.ordered())
        self.net.export(self.cfg.model_base, round)

    def params_file(self, round=None):
        if round is None: # Load the best params
            return f"{self.cfg.model_base}.params"
        else:
            return f"{self.cfg.model_base}-{round:04d}.params"
