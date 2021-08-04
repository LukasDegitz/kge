import time
import torch
import torch.utils.data
import math

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn
from kge.job.util import load_types


class TrainingJobHCMN(TrainingJob):
    """
        Implements hierarchical Multi-Label classification Network training as defined in Wehrmann et al. (2018)
        Codes are adapted from:
        https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/b4fa04ffb70cb4f3f2effdb07d455f5e3fc393ea/model/loss.py

    """
    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        config.log("Initializing HCMN training job...")
        self.type_str = "HCMN"
        if self.__class__ == TrainingJobHCMN:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()
        self.train_idx = self.model.type_ids['train']
        self.num_examples = len(self.train_idx)
        self.y = self.model.types
        self.hier = self.model.hier
        self.tuple_ids = self.model.get_tuple_ids()

        # loss function with positive weighting
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=self.model.pos_weights)

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
        train_idx = batch["idx"][subbatch_slice]
        train_idx = train_idx.to(self.device)
        lamb = self.model.get_lambda().to(self.device)
        y_train = batch["types"][subbatch_slice].float().to(self.device)
        self.loss.to(self.device)
        result.prepare_time += time.time()

        # forward pass
        result.forward_time = -time.time()
        (global_logits, local_logits, logits) = self.model.predict_all(train_idx, self.device)
        loss = self.loss(global_logits, y_train)
        loss += self.loss(local_logits, y_train)
        # penelize if child conf is bigger than parent conf, use confidence threshold 0.5
        if lamb > 0.00:
            y_hat = torch.sigmoid(logits)
            #lags parent prob to child
            y_parent = self.model.build_mask(y=y_hat, type='prob', device=self.device)
            max_pos = torch.nn.ReLU()

            violation_penalty = lamb * torch.sum(torch.pow(max_pos(y_hat-y_parent), 2))
            loss += violation_penalty.item()

        loss = self._loss_weight * loss / batch_size
        # losses
        result.avg_loss += loss.item()
        result.forward_time += time.time()

        # backward step for each local linear model
        result.backward_time = -time.time()
        if not self.is_forward_only:
            loss.backward()
        result.backward_time += time.time()
