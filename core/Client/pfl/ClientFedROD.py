import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F

from core.Client.ClientBase import Client


def balanced_softmax_loss(logits, labels, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss





class ClientFedROD(Client):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model, local_idx_dataidx_map, idx, logger, code_length, num_classes, device):
        super().__init__(args, model, local_idx_dataidx_map, idx, logger, code_length, num_classes, device)

    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.sgd
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)

        p_head_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.classifier.parameters()), lr=self.args.lr_sh_rate,
                                     )

        self.trainloader = self.get_trainloader()

        sample_per_class = self.generate_sample_per_class(self.trainloader)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                f, p = self.model(X)

                loss = balanced_softmax_loss(p, y, sample_per_class)

                loss = loss
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()

                logit_p = self.model.classifier(f.detach())
                logit = p.detach() + logit_p
                loss_p = F.cross_entropy(logit, y)
                loss_p.backward()
                p_head_optimizer.step()

                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)

    def generate_sample_per_class(self, local_data):
        sample_per_class = torch.tensor([0 for _ in range(self.num_classes)])

        for idx, (data, target) in enumerate(local_data):
            sample_per_class += torch.tensor([sum(target == i) for i in range(self.num_classes)])

        sample_per_class = torch.where(sample_per_class > 0, sample_per_class, 1)

        return sample_per_class