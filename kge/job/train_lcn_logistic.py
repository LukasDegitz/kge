import time
import torch
import torch.utils.data
import math

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn
from kge.job.util import load_types


class TrainingJobLCNLogisitic(TrainingJob):
    """ Samples types and optimizes entity embedding by logistic regression"""

    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        config.log("Initializing LCN logisitic training job...")
        self.type_str = "LCNLogistic"

        if self.__class__ == TrainingJobLCNLogisitic:
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

        # local loss function with manual loss weighting
        self.local_loss = torch.nn.BCELoss(reduction='none')

        # global loss function with positive weighting
        self.global_loss = torch.nn.BCEWithLogitsLoss(pos_weight=self.model.pos_weights)

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
        mask = self.model.get_train_mask(idx=train_idx).to(self.device)
        train_idx = train_idx.to(self.device)
        y_train = batch["types"][subbatch_slice].float().to(self.device)
        local = self.model.get_local().to(self.device)
        glob = self.model.get_global().to(self.device)
        self.local_loss.to(self.device)
        self.global_loss.to(self.device)
        result.prepare_time += time.time()

        # forward pass
        result.forward_time = -time.time()
        X_train = self.model.get_s_embedder().embed(train_idx)
        local_scores = local(X_train)
        local_scores = torch.sigmoid(local_scores) * mask
        global_scores = glob(torch.cat((X_train, local_scores), dim=1))

        # losses
        local_samples = (torch.sum(mask, dim=0) + 1) # plus 1 to avoid division by zero, loss is zero anyway
        local_loss = (torch.sum(self.local_loss(local_scores, y_train), dim=0) / local_samples).mean()
        global_loss = self.global_loss(global_scores, y_train)
        loss = self._loss_weight/batch_size * (local_loss + global_loss)

        result.avg_loss += loss.item()
        result.forward_time += time.time()

        # backward step for each local linear model
        result.backward_time = -time.time()
        if not self.is_forward_only:
            loss.backward()
        result.backward_time += time.time()


    def __train_lcn_logisitc_top_down(self, level, X_train, y_train, y_parent, result, y_scores, score_weights):

        # stop if hierarchy is passed
        if level >= len(self.hier):
            return y_scores, score_weights

        child_y_parent = {}
        for parent, children in self.hier[level].items():

            parent = str((level, parent))

            y_local_score = torch.zeros(len(y_train), dtype=torch.float).to(self.device)
            # skip if nothing to train
            if parent not in y_parent:
                y_scores.append(y_local_score)
                continue

            # keep track of allignment!!
            parent_type_id = self.tuple_ids[parent]
            parent_idx_to_pred = torch.nonzero(y_parent[parent])

            # slice local training data
            X_parent = X_train[parent_idx_to_pred]
            y_train_parent = y_train[parent_idx_to_pred, parent_type_id]

            result.forward_time = -time.time()
            parent_scores = self.model.score_type(type_tuple=parent, X=X_parent)
            y_local_score[parent_idx_to_pred] = parent_scores
            y_scores.append(y_local_score)
            score_weights.append(len(parent_idx_to_pred))
            # average loss per considered training entity

            # skip if no more positive examples left
            if sum(y_train_parent) == 0:
                continue

            #pass parent labels down to children
            for child in children:
                child = str((level + 1, child))
                # pass to every child, if parent was predicted. No tie Handling atm.
                if child not in child_y_parent:
                    child_y_parent[child] = y_train[:, parent_type_id]
                else:
                    child_y_parent[child] = torch.logical_or(child_y_parent[child],
                                                             y_train[:, parent_type_id]).int()

        return self.__train_lcn_logisitc_top_down(level=level + 1, X_train=X_train, y_train=y_train,
                                                  y_parent=child_y_parent, result=result, y_scores=y_scores,
                                                  score_weights=score_weights)

    def build_mask(self, y_train):
        mask = []
        y_parent = {}
        for root_type in self.hier[0].keys():
            y_parent[str((0, root_type))] = torch.ones(len(y_train), dtype=torch.int)
        for level, parents in self.hier.items():
            child_y_parent = {}
            for parent, children in parents.items():
                parent_tuple = str((level, parent))
                #if parent not in y_parent:
                #    mask.append(torch.zeros(len(y_train), dtype=torch.int))
                mask.append(y_parent[parent_tuple])
                parent_slice = self.tuple_ids[parent_tuple]
                for child in children:
                    child_tuple = str((level + 1, child))
                    if child_tuple not in child_y_parent:
                        child_y_parent[child_tuple] = y_train[:, parent_slice]
                    else:
                        child_y_parent[child_tuple] = torch.logical_or(child_y_parent[child_tuple],
                                                                       y_train[:, parent_slice]).int()
            y_parent = child_y_parent.copy()

        return torch.stack(mask).transpose(0, 1)
