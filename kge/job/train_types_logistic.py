import time
import torch
import torch.utils.data
import math

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn
from kge.job.util import load_types


class TrainingJobTypesLogisitic(TrainingJob):
    """ Samples types and optimizes entity embedding by logistic regression"""

    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        config.log("Initializing types logisitic training job...")
        self.type_str = "TypesLogistic"

        if self.__class__ == TrainingJobTypesLogisitic:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()
        # workaround for types dataset
        types_path = self.config.get('user.types_path')
        y, pos_weights, train_idx = load_types(types_path, self.dataset._num_entities, 'train')
        self.y = y
        self.train_idx = train_idx
        self.num_examples = len(train_idx)

        # overwrite loss function and add linear function for logistic
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        self.s_linear = torch.nn.Linear(in_features=self.model.get_s_embedder().dim, out_features=y.size(1), bias=False)
        torch.nn.init.constant_(self.s_linear.weight, math.sqrt(1 / self.model.get_s_embedder().dim))
        if not self.model.get_s_embedder() is self.model.get_o_embedder():
            self.o_linear = torch.nn.Linear(in_features=self.model.get_o_embedder().dim, out_features=y.size(1),
                                            bias=False)
            torch.nn.init.constant_(self.o_linear.weight, math.sqrt(1 / self.model.get_o_embedder().dim))
        # adapt loader, only take indices of training types
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=lambda batch: {
                "types": self.y[train_idx[batch], :],
                "idx": self.train_idx[batch]
            },
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        result.size = len(batch["idx"])

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):
        batch_size = result.size

        # prepare
        result.prepare_time = -time.time()
        types = batch["types"][subbatch_slice].to(self.device)
        train_idx = batch["idx"][subbatch_slice].to(self.device)
        self.s_linear.to(self.device)
        self.loss.to(self.device)
        result.prepare_time += time.time()

        # forward/backward pass (sp)
        result.forward_time = -time.time()
        X = self.model.get_s_embedder().embed(train_idx)
        scores = self.s_linear(X)
        s_loss = self._loss_weight * (self.loss(scores, types) / batch_size)
        result.avg_loss += s_loss.item()

        # take loss twice if same embedder for s and o
        if self.model.get_s_embedder() is self.model.get_o_embedder():
            result.avg_loss += s_loss.item()
            s_loss *= 2
            o_loss = torch.tensor([0.00])
        else:
            self.o_linear.to(self.device)
            X = self.model.get_o_embedder().embed(train_idx)
            scores = self.o_linear(X)
            o_loss = self._loss_weight * (self.loss(scores, types) / batch_size)
            result.avg_loss += o_loss.item()

        result.forward_time += time.time()
        result.backward_time = -time.time()
        if not self.is_forward_only:
            s_loss.backward()
            if not self.model.get_s_embedder() is self.model.get_o_embedder():
                o_loss.backward()
        result.backward_time += time.time()
