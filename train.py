from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
from network import Config, Network
import time

def iterative_training(network: Network):
    pass

def accuracy(output, label):
    return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()

class Training:
    def __init__(self, network: Network, cfg: Config):
        self.network = network
        self.epochs  = cfg.epochs
        self.batch_size  = cfg.batch_size
        self.params_file = cfg.params_file
        self.best_acc = 0.

    def train(self, dataset):
        train_data = gluon.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)
        valid_data = []
        for epoch in range(self.epochs):
            # Prepare data for one epoch
            train_loss, train_acc, valid_acc = 0., 0., 0.
            tick = time.time()

            for data, label in train_data:
                # predict on training set
                with autograd.record():
                    output = self.network.net(data)
                    loss = self.network.loss(output, label)
                # calculate gradients and update params
                loss.backward()
                self.network.trainer.step(self.batch_size)
                # collect loss and acc statistics
                train_loss += loss.mean().asscalar()
                train_acc += accuracy(output, label)

            for data, label in valid_data:
                valid_acc += accuracy(self.network.net(data), label)

            self.report_training(epoch,
                train_loss / len(train_data),
                train_acc / len(train_data),
                valid_acc / len(valid_data),
                time.time() - tick)

    # Log the training result, and update the model accordingly
    def report_training(self, epoch, loss, train_acc, valid_acc, time):
        # Log the result to console
        print(f"Epoch {epoch}: loss {loss:3f},",
            f"train acc {train_acc:3f},",
            f"valid acc {valid_acc:3f},"
            f"in {time:1f} sec")

        # Test whether to save the better one or reload the best
        if valid_acc >= self.best_acc:
            print(f"New best model found. Saving to {self.params_file}.")
            self.best_acc = valid_acc
            self.network.net.save_parameters(self.params_file)
        else:
            print(f"No better than best model. Reloading {self.params_file}.")
            self.network.cfg.learning_rate /= 4
            self.network.load_trainer()
            self.network.net.load_parameters(self.params_file)

        print("--------------------------------------------------")
