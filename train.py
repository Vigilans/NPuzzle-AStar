from AStar import Board, AStar
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
from network import Config, Network
import time

def accuracy(output, label):
    return (output == label.astype('float32')).mean().asscalar()

class Trainer:
    def __init__(self, network: Network):
        self.network = network
        self.train_proc = Training(network, network.cfg)
        self.generator = "manhattan"
        self.dataset_size = 1000
        self.max_steps = 5

    def generate_dataset(self, dataset_size: int):
        generator = AStar(self.generator)
        X, y = [], []
        while len(X) < dataset_size:
            board = Board.scrambled(self.max_steps, False)
            path, _, _, _ = generator.run(board)
            pathLengths, boardStates = list(zip(*enumerate(path)))
            X.extend(boardStates)
            y.extend(pathLengths)
        return gluon.data.ArrayDataset(nd.array(X), nd.array(y))

    def udpate(self):
        if type(self.generator) is str: # Currently cpp built-in for generating, update by imitation learning
            print(f"\nTesting network agent against {self.generator} agent...")
            agent_gen = AStar(self.generator)
            agent_net = AStar(self.network.predict)
            result_gen = []
            result_net = []
            for i in range(5): # test 5 rounds
                board = Board.scrambled(self.max_steps, True) # fixed length
                result_gen.append(agent_gen.run(board))
                result_net.append(agent_net.run(board))
                print(f"Round {i + 1}, gen: {result_gen[-1][1:]}")
                print(f"Round {i + 1}, net: {result_net[-1][1:]}")

            if self.max_steps < 30:
                # We use explorered states to determine whether max_step should be higher
                avg_states_gen = sum([r[2] for r in result_gen]) / 5
                avg_states_net = sum([r[2] for r in result_net]) / 5
                print(f"Explorered states: gen({avg_states_gen}) vs net({avg_states_net})")
                if avg_states_net < avg_states_gen:
                    self.max_steps += 5
                    print(f"Max scrambling steps increased to {self.max_steps}")
            else: # Bound to turn imitation learning to curriculumn learning
                # We use running time to determine whether it is suitable to use network to predict
                avg_time_gen = sum([r[3] for r in result_gen]) / 5
                avg_time_net = sum([r[3] for r in result_net]) / 5
                print(f"Running time: gen({avg_time_gen}) vs net({avg_time_net})")
                if avg_time_net < avg_time_gen:
                    print("Generator changed to network.")
                    self.generator = self.network.predict
                else:
                    print("Generator keeps unchanged.")
        else: # Currently network.predict for generating, update by curriculum learning
            print(f"\nTesting network agent on random state predicting accuracy...")
            # Generate a test set to see the accuracy
            testset = self.generate_dataset(10)
            states, labels = list(zip(*testset))
            acc = accuracy(self.network.net(states), labels)
            print(f"Accuracy: {acc}")
            if (acc > 0.75):
                self.max_steps += 2
                print(f"Max scrambling steps increased to {self.max_steps}")

    def start(self):
        while True:
            dataset = self.generate_dataset(self.dataset_size)
            self.train_proc.train(dataset)
            self.udpate()

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
