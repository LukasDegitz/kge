import torch
import time
import math

from kge import Config, Dataset
from kge.job import Job, EvaluationJob
from kge.job.util import load_types

from typing import Any, Dict


class TypesLogisticEvaluationJob(EvaluationJob):
    """ Evaluating by using the training loss """

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)

        config.log("Initializing types logisitic validation job...")
        self.type_str = "TypesLogistic"

        # add MRR eval job
        entity_ranking_eval_config = self.config.clone()
        entity_ranking_eval_config.set('eval.type', 'entity_ranking')
        #entity_ranking_eval_config.set('console.quiet', True)
        self.mrr_job = EvaluationJob.create(entity_ranking_eval_config,
                                            dataset=self.dataset,
                                            parent_job=self,
                                            model=self.model)

        if self.__class__ == TypesLogisticEvaluationJob:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        super()._prepare()
        self.mrr_job._prepare()
        """Construct dataloader"""
        # workaround for types dataset
        types_path = self.config.get('user.types_path')
        y, pos_weights, valid_idx = load_types(types_path, self.dataset._num_entities, self.eval_split)
        self.y = y
        self.valid_idx = valid_idx
        self.num_examples = len(valid_idx)

        # overwrite loss function and add linear function for logistic
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        self.s_linear = torch.nn.Linear(in_features=self.model.get_s_embedder().dim, out_features=y.size(1), bias=False)
        torch.nn.init.constant_(self.s_linear.weight, math.sqrt(1 / self.model.get_s_embedder().dim))
        if not self.model.get_s_embedder() is self.model.get_o_embedder():
            self.o_linear = torch.nn.Linear(in_features=self.model.get_o_embedder().dim, out_features=y.size(1),
                                            bias=False)
            torch.nn.init.constant_(self.o_linear.weight, math.sqrt(1 / self.model.get_o_embedder().dim))
        # and data loader
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=lambda batch: {
                "types": self.y[valid_idx[batch], :],
                "idx": self.valid_idx[batch]
            },
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),
        )


    @torch.no_grad()
    def _evaluate(self) -> Dict[str, Any]:
        if self.parent_job:
            self.epoch = self.parent_job.epoch

        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type="types_logistic_evaluation",
            scope="epoch",
            split=self.eval_split,
            epoch=self.epoch,
            batches=len(self.loader),
            size=self.num_examples,
        )

        #lets go
        epoch_time = -time.time()
        avg_loss = 0
        for i, batch in enumerate(self.loader):

            self.current_trace["batch"] = dict(
                type="types_logistic_evaluation",
                scope="batch",
                split=self.eval_split,
                epoch=self.epoch,
                batch=i,
                size=len(batch["idx"]),
                batches=len(self.loader),
            )

            types = batch["types"].to(self.device)
            valid_idx = batch["idx"].to(self.device)
            self.s_linear.to(self.device)
            self.loss.to(self.device)

            X = self.model.get_s_embedder().embed(valid_idx)
            scores = self.s_linear(X)
            s_loss = self.loss(scores, types) / len(valid_idx)
            avg_loss += s_loss.item()
            if self.model.get_s_embedder() is self.model.get_o_embedder():
                avg_loss += s_loss.item()
                s_loss *= 2
                o_loss = torch.tensor([0.00])
            else:
                self.o_linear.to(self.device)
                X = self.model.get_o_embedder().embed(valid_idx)
                scores = self.o_linear(X)
                o_loss = self.loss(scores, types) / len(valid_idx)
                avg_loss += o_loss.item()
            # update batch trace with the results
            self.current_trace["batch"].update(dict(
                loss=s_loss.item()+o_loss.item(),
            ))

            # run the post-batch hooks (may modify the trace)
            for f in self.post_batch_hooks:
                f(self)

            # output, then clear trace
            if self.trace_batch:
                self.trace(**self.current_trace["batch"])
            self.current_trace["batch"] = None

            # output batch information to console
            self.config.print(
                (
                    "\r"  # go back
                    + "{}  batch:{: "
                    + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                    + "d}/{}, avg loss: {:4.3f}"
                    + "\033[K"  # clear to right
                ).format(
                    self.config.log_prefix,
                    i,
                    len(self.loader) - 1,
                    avg_loss,
                ),
                end="",
                flush=True,
            )


        self.mrr_job.epoch = self.epoch
        self.mrr_job._evaluate()

        self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back
        epoch_time += time.time()
        # compute trace
        self.current_trace["epoch"].update(self.mrr_job.current_trace['epoch'])
        self.current_trace["epoch"].update(
            dict(
                epoch_time=epoch_time,
                event="eval_completed",
                avg_loss=avg_loss
            )
        )
        self.mrr_job.current_trace["epoch"] = None