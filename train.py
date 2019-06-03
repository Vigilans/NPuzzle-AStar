from AStar import Board, AStar
from mxnet import nd, gluon, init, autograd, gpu
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
from network import Config, Network
from datetime import timedelta
import numpy as np
import time

def accuracy(output, label):
    return (output.round() == label).mean().asscalar()

class Trainer:
    def __init__(self, network: Network):
        self.network = network
        self.train_proc = Training(network, network.cfg)
        self.generator = "manhattan"
        self.dataset_size = 5000
        self.max_steps = 5

    def generate_dataset(self, dataset_size: int):
        generator = AStar(self.generator)
        X, y = [Board.ordered().state], [[0]]
        while len(X) < dataset_size:
            boards = Board.scrambled(self.max_steps, True)
            path, pathLength, _, _ = generator.run(boards[-1])
            for i, board in enumerate(path[:-1]):
                X.append(board.state)
                y.append([pathLength - 1 - i]) # to ensure consistent shape
        X = [self.network.transform(state) for state in X]
        return X, y

    def update(self):
        if type(self.generator) is str: # Currently cpp built-in for generating, update by imitation learning
            print(f"\nTesting network agent against {self.generator} agent...")
            agent_gen = AStar(self.generator)
            agent_net = AStar(self.network.predict)
            result_gen = []
            result_net = []
            for i in range(5): # test 5 rounds
                boards = Board.scrambled(self.max_steps, True) # fixed length
                result_gen.append(agent_gen.run(boards[-1]))
                print(f"Round {i + 1}, gen: {result_gen[-1][1:]}")
                result_net.append(agent_net.run(boards[-1]))
                print(f"Round {i + 1}, net: {result_net[-1][1:]}")

            if self.max_steps < 30:
                # We use explorered states to determine whether max_step should be higher
                avg_states_gen = sum([r[2] for r in result_gen]) / 5
                avg_states_net = sum([r[2] for r in result_net]) / 5
                print(f"Explorered states: gen({avg_states_gen}) vs net({avg_states_net})")
                if avg_states_net < avg_states_gen:
                    self.max_steps += (5 if self.max_steps < 25 else 1)
                    print(f"Max scrambling steps increased to {self.max_steps}")
                    return True
            else: # Bound to turn imitation learning to curriculumn learning
                # We use running time to determine whether it is suitable to use network to predict
                avg_time_gen = sum([r[3] for r in result_gen], timedelta()) / 5
                avg_time_net = sum([r[3] for r in result_net], timedelta()) / 5
                print(f"Running time: gen({avg_time_gen}) vs net({avg_time_net})")
                if avg_time_net < avg_time_gen:
                    print("Generator changed to network.")
                    self.generator = self.network.predict
                    return True
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
                return True
        print() # Leave an empty line
        return False # Did not update

    def start(self):
        while True:
            while self.update():
                continue # continue increasing difficulty until need more training
            print(f"Generating {self.dataset_size + self.dataset_size / 10} examples...")
            train_set = self.generate_dataset(self.dataset_size)
            valid_set = self.generate_dataset(self.dataset_size / 10)
            self.train_proc.train(train_set, valid_set)
            self.network.net.load_parameters(self.network.cfg.params_file, ctx=gpu(0)) # reload best model

class Training:
    def __init__(self, network: Network, cfg: Config):
        self.network = network
        self.epochs  = cfg.epochs
        self.batch_size  = cfg.batch_size
        self.params_file = cfg.params_file
        self.best_acc = float("inf")

    def build_dataset(self, dataset):
        # states = dataset[0]
        # labels = dataset[1]
        # sampler = gluon.data.RandomSampler(len(states))
        # sampler = gluon.data.BatchSampler(sampler, self.batch_size, "discard")
        # state_batches = []
        # label_batches = []
        # print(f"Generating batches of epoch times for dataset length {len(states)}")
        # for _ in range(self.epochs):
        #     state_batches.append([[states[i] for i in batch] for batch in sampler])
        #     label_batches.append([[labels[i] for i in batch] for batch in sampler])
        # print(f"Putting dataset to GPUs...")
        # state_batches = nd.array(state_batches, ctx=gpu(0))
        # label_batches = nd.array(label_batches, ctx=gpu(0))
        # def yield_example(epoch):
        #     for i in range(len(sampler)):
        #         yield state_batches[epoch][i], label_batches[epoch][i]
        # return yield_example, len(sampler)
        dataset = gluon.data.ArrayDataset(nd.array(dataset[0]), nd.array(dataset[1]))
        loader = gluon.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        def yield_example(epoch):
            for state, label in loader:
                yield state.as_in_context(gpu(0)), label.as_in_context(gpu(0))
        return yield_example, len(loader)

    def train(self, train_set, valid_set):
        print("Building datasets...")
        train_data, train_batches = self.build_dataset(train_set)
        valid_data, valid_batches = self.build_dataset(valid_set)
        print("Start training...")
        for epoch in range(self.epochs):
            # Prepare data for one epoch
            train_loss, train_acc, valid_acc = 0., 0., 0.
            tick = time.time()

            for data, label in train_data(epoch):
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

            for data, label in valid_data(epoch):
                valid_acc += accuracy(self.network.net(data), label)

            self.report_training(epoch,
                train_loss / train_batches,
                train_acc / train_batches,
                valid_acc / valid_batches,
                time.time() - tick)

    # Log the training result, and update the model accordingly
    def report_training(self, epoch, loss, train_acc, valid_acc, time):
        # Log the result to console
        print(f"Epoch {epoch}: loss {loss:3f}, "
            f"train acc {train_acc:3f}, "
            f"valid acc {valid_acc:3f}, "
            f"in {time:1f} sec")

        # Test whether to save the better one or reload the best
        if self.best_acc == float("inf"):
            self.best_acc = valid_acc
        if valid_acc > self.best_acc:
            print(f"New best model found. Saving to {self.params_file}.")
            print("--------------------------------------------------")
            self.best_acc = valid_acc
            self.network.net.save_parameters(self.params_file)
        else:
            # self.network.cfg.learning_rate /= 4
            # self.network.load_trainer()
            # self.network.net.load_parameters(self.params_file, ctx=gpu(0))
            pass
