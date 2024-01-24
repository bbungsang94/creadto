import copy
import math
import os
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from utils.io import clean_folder, make_dir, save_torch
from utils.log import CSVWriter, print_message
from utils.report import get_input_variable_count


class Base(metaclass=ABCMeta):
    def __init__(self, viewer=None,
                 loaders: Tuple[DataLoader, DataLoader] = None,
                 model: nn.Module = None,
                 loss: nn.Module = None,
                 metric: nn.Module = None,
                 optimizer: Optimizer = None,
                 params: dict = None
                 ):
        self._viewer = viewer
        self._loader, self._evaluator = loaders
        self._model = model
        self._loss = loss
        if metric:
            self._metric = metric
        else:
            self._metric = loss
        self._optimizer = optimizer
        self._params = params

        # base
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        log_path = os.path.join(self._params['path']['log'], self._params['task']['model_name'], timestamp)
        clean_folder(os.path.join(self._params['path']['log'], self._params['task']['model_name']))
        self._writer = CSVWriter(log_path)
        self.name = 'base'
        self.model_name = self._params['task']['model_name']

        # device
        self.device = torch.device("cpu")

        # inner
        self.__loop = self.loop
        self.best_loss = math.inf

    @abstractmethod
    def report_state(self):
        pass

    @abstractmethod
    def _write_log(self, **kwargs):
        pass

    def _check_sanity(self):
        make_dir(os.path.join(self._params['path']['checkpoint'], self.model_name))
        make_dir(os.path.join(self._params['path']['log'], self.model_name))
        print(print_message(message='', line='='))
        print(print_message(message='Checking Sanity', padding=3, center=True, line='-'))
        print(print_message(message=''))
        model_input = get_input_variable_count(self._model.forward)
        # loader_output = get_return_variable_count(self._loader.collate_fn)
        loader_output = 2
        print(print_message(message='Loader output: ' + str(loader_output), padding=2))
        print(print_message(message='Model Input: ' + str(model_input), padding=2))
        # model_output = get_return_variable_count(self._model.forward)
        print(print_message(message='Model output: ' + str(2), padding=2))
        epoch = 0
        tick = 0
        if self._params['task']['resume']:
            print(print_message(message='Resume Process', padding=3, center=True, line='-'))
            epochs = os.listdir(os.path.join(self._params['path']['checkpoint'], self.model_name))
            if len(epochs) == 0:
                print(
                    "'\033[91m" + "Not found: can't find checkpoints of this model. Canceled the resuming" + "\033[0m")
            else:
                if "Best" in epochs:
                    # Update loss from the best model
                    weights = torch.load(os.path.join(self._params['path']['checkpoint'],
                                                      self.model_name, "Best", "BestModel.pth"))
                    # self.best_loss = weights['loss']
                    self.best_loss = 0.00002

                epochs = [int(x) for x in epochs if x != "Best"]
                epoch = max(epochs)
                ticks = os.listdir(os.path.join(self._params['path']['checkpoint'], self.model_name, "%08d" % epoch))
                ticks = [int(x) for x in ticks]
                tick = max(ticks)
                full_path = os.path.join(self._params['path']['checkpoint'], self.model_name, "%08d" % epoch,
                                         "%08d" % tick)
                model_file = os.listdir(full_path)[0]
                self._load_model(os.path.join(full_path, model_file))
                print(print_message(message='Epoch: ' + str(epoch), padding=2))
                print(print_message(message='Tick: ' + str(tick), padding=2))
                print(print_message(message='Model Name: ' + model_file, padding=2))

        self._params['task']['itr'] = [epoch, self._params['hyperparameters']['epochs']]
        self._params['task']['tick'] = tick
        if (loader_output - 1) != model_input:
            print("'\033[91m" + "CONFLICT: Different loader output with model input shapes" + "\033[0m")
        if "sample" in dir(self._loader) and self._params['task']['visualize']:
            self._viewer.show(**self._loader.sample())

    def _save_model(self, epoch: int, tick: int, prefix='', loss=None):
        timeline = prefix + datetime.now().strftime('%Y%m%d%H%M%S') + '.pth'
        mode = "state_dict"
        if tick < 0:
            full_path = os.path.join(self._params['path']['checkpoint'], self.model_name, "Best")
            timeline = timeline.replace(".pth", "-BestModel.pth")
            # mode = "jit"
        else:
            full_path = os.path.join(self._params['path']['checkpoint'], self.model_name, "%08d" % epoch, "%08d" % tick)
        if not os.path.exists(full_path):
            os.mkdir(full_path)

        loss = loss if loss is not None else 999.9
        weights = {'epochs': epoch,
                   'loss': loss}
        if "state_dict" in dir(self._loader):
            weights.update(self._loader.state_dict())
        save_torch(os.path.join(full_path, timeline), model=self._model, mode=mode, **weights)

    def _load_model(self, full_path):
        weights = torch.load(full_path)
        if "load_state_dict" in dir(self._loader):
            self._loader.load_state_dict(weights)
        self._model.load_state_dict(weights['model'])

    def loop(self) -> None:
        self._check_sanity()
        for epoch in range(*self._params['task']['itr']):
            if not os.path.exists(os.path.join(self._params['path']['checkpoint'], self.model_name, "%08d" % epoch)):
                os.mkdir(os.path.join(self._params['path']['checkpoint'], self.model_name, "%08d" % epoch))

            # train
            train_pbar = tqdm(self._loader, desc='Train', position=0, leave=True, ncols=80)
            if self._params['task']['tick'] > 0:
                train_pbar.total -= copy.deepcopy(self._params['task']['tick'])
                self._params['task']['tick'] = 0
            train_mu = self._run_train_epoch(epoch, train_pbar)

            # evaluation
            eval_pbar = tqdm(self._evaluator, desc='Eval', position=0, leave=True, ncols=80)
            eval_mu = self._run_eval_epoch(epoch, eval_pbar)

            # recording
            self._writer.add_scalars('Training vs Evaluation Loss',
                                     {'Epoch': epoch, 'Training': train_mu, 'Evaluation': eval_mu})
            self._writer.flush()

            # update best model
            self.__update_best(loss=eval_mu, epoch=epoch)

    def __update_best(self, loss, epoch):
        if self.best_loss > loss:
            self.best_loss = loss
            print("Updated best model.")
            self._save_model(epoch, -1, loss=self.best_loss)
            if "summary" in dir(self._model) and "sample" in dir(self._loader):
                result = self._model.summary(**self._loader.sample())
                result['save_path'] = os.path.join(self._params['path']['checkpoint'], self.model_name, "Best")
                result['model'] = self._model
                self._viewer.summary(**result)

    def _run_train_epoch(self, index, progress):
        self._model.train()
        running_loss = 0.
        avg_loss = 0
        mode = progress.desc
        line = "Calculating loss..."
        progress.set_description(line)
        for i, data in enumerate(progress):
            if progress.total < progress.n:
                return 1.0

            inputs, labels = data
            output = self._model(inputs)

            self._optimizer.zero_grad()
            if not isinstance(labels, torch.Tensor):
                loss = self._loss(output['output'][0], labels[0])
                for itr in range(1, len(labels)):
                    loss += self._loss(output['output'][itr], labels[itr])
            else:
                loss = self._loss(output['output'], labels)
            loss.backward()
            self._optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
            line = "avg_loss: %.4f, ticks: %06d" % (avg_loss, i)
            progress.set_description(line)

            if i % self._params['task']['log_interval'] == self._params['task']['log_interval'] - 1:
                save_path = os.path.join(self._params['path']['checkpoint'],
                                         self.model_name, "%08d" % index, "%08d" % i)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                viewer_kwargs = {'inputs': inputs[0],
                                 'labels': labels,
                                 'latent': output['latent'],
                                 'outputs': output['output'][0]}
                self._viewer.save(images=viewer_kwargs, save_path=save_path)
                avg_loss = running_loss / i  # loss per batch
                self._write_log(epoch=index, tick=i, loss=avg_loss, mode=mode)
                self._save_model(index, i, loss=avg_loss)

        return avg_loss

    def _run_eval_epoch(self, index, progress):
        self._model.eval()
        running_loss = 0.
        avg_loss = 0.
        with torch.no_grad():
            for i, data in enumerate(progress):
                inputs, labels = data
                output = self._model(inputs)
                if not isinstance(labels, torch.Tensor):
                    loss = self._metric(output['output'][0], labels[0])
                    for itr in range(1, len(labels)):
                        loss += self._metric(output['output'][itr], labels[itr])
                else:
                    loss = self._metric(output['output'], labels)
                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
                line = "evaluation loss: %.4f, ticks: %06d" % (avg_loss, i)
                progress.set_description(line)

            return avg_loss
