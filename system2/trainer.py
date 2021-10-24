from io import TextIOWrapper
from sys import stdout
from time import time
from typing import Any, Dict, List, Tuple, Type

import torch
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.optim import lr_scheduler as sched

from board import SummaryWriter, get_writer
from loader import DataLoader
from model import Model, nn
from util import LOG_DIR, MODEL_DIR, Path


class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer: Type[optim.Optimizer], rho: float = 0.05,
                 **kwargs) -> None:
        super(SAM, self).__init__(params, dict(rho=rho, **kwargs))
        self.base_optimizer: optim.Optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups: List[Dict] = self.base_optimizer.param_groups

    @torch.no_grad()
    def step_1(self, zero_grad: bool = False) -> None:
        norm: torch.FloatTensor = self._grad_norm()

        for group in self.param_groups:
            scale: torch.FloatTensor = group['rho'] / (norm + 1e-8)
            param: torch.FloatTensor

            for param in group['params']:
                if param.grad is None:
                    continue

                eps: torch.FloatTensor = param.grad * scale.to(param)
                param.add_(eps)
                self.state[param]['eps'] = eps

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step_2(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            param: torch.FloatTensor

            for param in group['params']:
                if param.grad is not None:
                    param.sub_(self.state[param]['eps'])

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.FloatTensor:
        device: torch.device = self.param_groups[0]['params'][0].device
        return torch.norm(torch.stack([param.grad.norm(p=2).to(device)
                                       for group in self.param_groups for param in group['params']
                                       if param.grad is not None]), p=2)


class Trainer:
    def __init__(self, name: str, model: Model, optim_params: Dict[str, float]) -> None:
        self.model: Model = model
        self.model_path: Path = MODEL_DIR.joinpath(f'{name}.pt')
        self.log_path: Path = LOG_DIR.joinpath(f'{name}.log')
        self.writer: SummaryWriter = get_writer(name)

        learning_rate: float = optim_params['learning_rate']
        self.optimizer: SAM = SAM(self.model.parameters(), optim.Adam, lr=learning_rate,
                                  weight_decay=optim_params['weight_decay'])
        self.scheduler: sched.StepLR = sched.StepLR(
            self.optimizer.base_optimizer, optim_params['step_size'], optim_params['lr_decay']
        )

        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.gradient_clip: float = optim_params['gradient_clip']
        print(f'Initial learning rate: {learning_rate:.1e}\tGradient clip: {self.gradient_clip}')

    def train(self, train: DataLoader, valid: DataLoader, test: DataLoader, patience: int,
              max_epoch: int) -> None:
        start_time: float = time()
        self.log_io: TextIOWrapper = self.log_path.open('w', encoding='utf8')

        best_loss: float = self._evaluate(valid)[0]
        best_state: Dict[str, Any] = self.model.state_dict()
        torch.save(best_state, self.model_path)

        self.epoch: int = 0
        patience_count: int = 0

        if max_epoch <= 0:
            max_epoch = -1
        if patience <= 0:
            patience = -1

        for f in (stdout, self.log_io):
            print(f'Patience: {patience} Max epoch: {max_epoch}', file=f)
            f.flush()

        while patience_count != patience and self.epoch != max_epoch:
            epoch_start_time: float = time()

            train_loss: float
            train_acc: float
            train_loss, train_acc = self._train_epoch(train)

            valid_loss: float
            valid_acc: float
            macro_acc: float
            macro_p: float
            macro_r: float
            macro_f: float
            valid_loss, valid_acc, macro_acc, macro_p, macro_r, macro_f = self._evaluate(valid)[:6]

            self.writer.add_scalars('Loss', {'Train': train_loss, 'Valid': valid_loss}, self.epoch)
            self.writer.add_scalars('Accuracy', {'Train': train_acc, 'Valid': valid_acc},
                                    self.epoch)
            self.writer.add_scalars('Macro Metrics', {'Accuracy': macro_acc, 'Precision': macro_p,
                                                      'Recall': macro_r, 'F-1': macro_f},
                                    self.epoch)
            self.writer.flush()
            elapsed_time: float = time() - epoch_start_time

            for f in (stdout, self.log_io):
                print(f'Epoch: {self.epoch:2d}\tElapsed time: {elapsed_time:7.2f}s\t' +
                      f'Train loss: {train_loss:7.4f}\tTrain accuracy: {train_acc:7.2%}\t' +
                      f'Valid loss: {valid_loss:7.4f}\tValid accuracy: {valid_acc:7.2%}\t' +
                      f'Valid macro accuracy: {macro_acc:7.2%}\t' +
                      f'Valid macro precision: {macro_p:7.2%}\t' +
                      f'Valid macro recall: {macro_r:7.2%}\tValid macro F-1: {macro_f:7.4f}',
                      file=f)
                f.flush()

            if patience == -1 or valid_loss < best_loss:
                best_loss = valid_loss
                best_state = self.model.state_dict()
                torch.save(best_state, self.model_path)
                patience_count = 0
            else:
                patience_count += 1

        self.model.load_state_dict(best_state)

        test_loss: float
        test_acc: float
        macro_acc: float
        macro_p: float
        macro_r: float
        macro_f: float
        test_auc: float
        class_acc: torch.FloatTensor
        class_p: torch.FloatTensor
        class_r: torch.FloatTensor
        class_f: torch.FloatTensor
        (test_loss, test_acc, macro_acc, macro_p, macro_r, macro_f, test_auc, class_acc, class_p,
         class_r, class_f) = self._evaluate(test)
        total_elapsed_time: float = time() - start_time

        for f in (stdout, self.log_io):
            print(f'Total elapsed time: {total_elapsed_time:7.2f}s\tTest loss: {test_loss:7.4f}\t' +
                  f'Test accuracy: {test_acc:7.2%}\tTest macro accuracy: {macro_acc:7.2%}\t' +
                  f'Test macro precision: {macro_p:7.2%}\tTest macro recall: {macro_r:7.2%}\t' +
                  f'Test macro F-1: {macro_f:7.4f}\tTest AUC: {test_auc:7.4f}', file=f)

            for i in range(4):
                print(f'Type {i} accuracy: {class_acc[i].item():7.2%}\t' +
                      f'Type {i} precision: {class_p[i].item():7.2%}\t' +
                      f'Type {i} recall: {class_r[i].item():7.2%}\t' +
                      f'Type {i} F-1: {class_f[i].item():7.4f}', file=f)

        self.log_io.close()

    def _train_epoch(self, data: DataLoader) -> Tuple[float, float]:
        total_loss: int = 0
        total_correct: int = 0
        total_count: int = 0
        self.model.train()

        total_iter: int = len(data)
        verbose: List[int] = [round(total_iter / 10 * (i + 1)) for i in range(10)]

        cur_elapsed_time: float = 0
        cur_total_iter: int = 0
        cur_total_loss: float = 0
        cur_total_correct: int = 0
        cur_total_count: int = 0

        user_data: torch.FloatTensor
        tweet_data: torch.FloatTensor
        user_type: torch.LongTensor
        index: int = 0

        for user_data, tweet_data, user_type in data:
            batch_size: int = user_type.size(0)
            # limit: int = batch_size if self.epoch < 10 else batch_size >> 2
            limit: int = batch_size
            start_time: float = time()
            self.optimizer.zero_grad()

            pred_type: torch.FloatTensor = self.model.forward(user_data, tweet_data)
            loss: torch.FloatTensor = self.criterion.forward(pred_type, user_type)
            hard_pos: torch.LongTensor = loss.argsort(descending=True)[:limit]
            loss = loss[hard_pos].mean()
            loss.backward()
            nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step_1(zero_grad=True)

            self.criterion.forward(
                self.model.forward(user_data[hard_pos], tweet_data[hard_pos]), user_type[hard_pos]
            ).mean().backward()
            nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step_2(zero_grad=True)

            index += 1
            cur_total_iter += 1
            cur_total_count += batch_size
            cur_total_loss += loss.item() * batch_size
            cur_total_correct += torch.sum(pred_type.argmax(dim=1) == user_type).item()
            cur_elapsed_time += (time() - start_time) * 1000

            if index in verbose:
                progress: float = index / total_iter
                loss_mean: float = cur_total_loss / cur_total_count
                accuracy: float = cur_total_correct / cur_total_count
                time_mean: float = cur_elapsed_time / cur_total_iter

                print(f'Iter: {index:5d}\tProgress: {progress:4.0%}\tLoss: {loss_mean:7.4f}\t' +
                      f'Accuracy: {accuracy:7.2%}\tSpeed: {time_mean:7.2f} ms/iter',
                      file=self.log_io)
                self.log_io.flush()

                total_count += cur_total_count
                total_loss += cur_total_loss
                total_correct += cur_total_correct
                cur_elapsed_time = cur_total_loss = 0
                cur_total_iter = cur_total_correct = cur_total_count = 0

        self.scheduler.step()
        self.epoch += 1
        return total_loss / total_count, total_correct / total_count

    def _evaluate(self, data: DataLoader) -> Tuple[float, float, float, float, float, float, float,
                                                   torch.FloatTensor, torch.FloatTensor,
                                                   torch.FloatTensor, torch.FloatTensor]:
        self.model.eval()
        total_loss: float = 0

        true_positive: torch.LongTensor = torch.zeros(4, dtype=torch.long)
        false_positive: torch.LongTensor = torch.zeros(4, dtype=torch.long)
        true_negative: torch.LongTensor = torch.zeros(4, dtype=torch.long)
        false_negative: torch.LongTensor = torch.zeros(4, dtype=torch.long)

        pred_type_list: List[torch.FloatTensor] = []
        user_type_list: List[torch.LongTensor] = []

        user_data: torch.FloatTensor
        tweet_data: torch.FloatTensor
        user_type: torch.LongTensor

        with torch.no_grad():
            for user_data, tweet_data, user_type in data:
                pred_type: torch.FloatTensor = self.model.forward(user_data, tweet_data)
                pred_type_list.append(pred_type)
                user_type_list.append(user_type)

                loss: torch.FloatTensor = self.criterion.forward(pred_type, user_type).mean()
                total_loss += loss.item() * user_data.size(0)
                pred_label: torch.LongTensor = pred_type.argmax(dim=1)

                for i in range(4):
                    true_positive[i] += torch.sum(user_type[pred_label == i] == i).cpu()
                    false_positive[i] += torch.sum(user_type[pred_label == i] != i).cpu()
                    true_negative[i] += torch.sum(user_type[pred_label != i] != i).cpu()
                    false_negative[i] += torch.sum(user_type[pred_label != i] == i).cpu()

        all_pred_type: torch.FloatTensor = torch.softmax(torch.cat(pred_type_list), dim=1)
        all_user_type: torch.LongTensor = torch.cat(user_type_list)
        auc: float = roc_auc_score(all_user_type.cpu().numpy(), all_pred_type.cpu().numpy(),
                                   multi_class='ovo')

        count: torch.LongTensor = true_positive + false_positive + true_negative + false_negative
        correct: torch.LongTensor = true_positive + true_negative
        predictions: torch.LongTensor = true_positive + false_positive
        labels: torch.LongTensor = true_positive + false_negative

        accuracy: torch.FloatTensor = correct / count
        precision: torch.FloatTensor = true_positive / predictions
        recall: torch.FloatTensor = true_positive / labels
        f_1: torch.FloatTensor = 2 * precision * recall / (precision + recall)

        macro_accuracy: float = accuracy.mean().item()
        macro_precision: float = precision.mean().item()
        macro_recall: float = recall.mean().item()
        macro_f_1: float = f_1.mean().item()

        total_correct: int = true_positive.sum().item()
        total_count: int = total_correct + false_positive.sum().item()
        average_loss: float = total_loss / total_count
        micro_f_1: float = total_correct / total_count

        return (average_loss, micro_f_1, macro_accuracy, macro_precision, macro_recall, macro_f_1,
                auc, accuracy, precision, recall, f_1)
