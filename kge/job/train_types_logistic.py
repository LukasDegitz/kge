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
        self.train_idx = self.model.type_ids['train']
        self.num_examples = len(self.train_idx)
        self.y = self.model.types

        # overwrite loss function and add linear function for logistic
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=self.model.types_loss_weights)
        # adapt loader, only take indices of training types
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=lambda batch: {
                "types": self.y[self.train_idx[batch], :],
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
        linear = self.model.get_linear()
        self.loss.to(self.device)
        result.prepare_time += time.time()

        # forward/backward pass (sp)
        result.forward_time = -time.time()
        X = self.model.get_s_embedder().embed(train_idx)
        scores = linear(X)
        loss = self._loss_weight * (self.loss(scores, types) / batch_size)
        result.avg_loss += loss.item()

        result.forward_time += time.time()
        result.backward_time = -time.time()
        if not self.is_forward_only:
            loss.backward()
        result.backward_time += time.time()
