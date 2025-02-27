import torch
import time
import math
import numpy as np

from kge import Config, Dataset
from kge.job import Job, EvaluationJob

from typing import Any, Dict
from sklearn.metrics import f1_score


class HMCNEvaluationJob(EvaluationJob):
    """ Evaluating by using hierarchical type prediction model by average f1"""

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)

        config.log("Initializing HMCN validation job...")

        # add MRR eval job
        entity_ranking_eval_config = self.config.clone()
        entity_ranking_eval_config.set('eval.type', 'entity_ranking')
        #entity_ranking_eval_config.set('console.quiet', True)
        self.mrr_job = EvaluationJob.create(entity_ranking_eval_config,
                                            dataset=self.dataset,
                                            parent_job=self,
                                            model=self.model)

        # Set static confidence threshold
        # TODO: implement one threshold per class and corresponding optimization e.g. Pellegrini and Masquelier (2021)
        self.threshold = torch.Tensor([0.5])

        if self.__class__ == HMCNEvaluationJob:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        super()._prepare()
        self.mrr_job._prepare()
        """Construct dataloader"""
        # workaround for types dataset
        self.valid_idx = self.model.type_ids['valid']
        self.num_examples = len(self.valid_idx)
        self.y = self.model.types


        # and data loader
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=lambda batch: {
                "types": self.y[self.valid_idx[batch], :],
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
        mean_f1 = 0
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

            y_valid = batch["types"].to(self.device)
            valid_idx = batch["idx"].to(self.device)
            thresh = self.threshold.to(self.device)

            (_, _, probits) = self.model.predict_all(idx=valid_idx, device=self.device)
            y_hat = torch.where(probits > thresh, 1, 0)

            # mask inconsistent predictions -> penalized if HMCN.lamb > 0
            y_hat_mask = self.model.build_mask(y_hat, type='binary', device=self.device)
            y_hat = y_hat*y_hat_mask

            f1 = f1_score(y_valid.cpu().numpy(), y_hat.cpu().numpy(), average='micro', zero_division=0)
            mean_f1 += f1 / len(self.loader)
            # update batch trace with the results
            self.current_trace["batch"].update(dict(
                f1=f1,
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
                    + "d}/{}, avg_f1 = {:.5f}"
                    + "\033[K"  # clear to right
                ).format(
                    self.config.log_prefix,
                    i,
                    len(self.loader) - 1,
                    f1,
                ),
                end="",
                flush=True,
            )

        mean_f1 = float(mean_f1)
        self.mrr_job.epoch = self.epoch
        self.mrr_job._evaluate()

        self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back
        epoch_time += time.time()
        # compute trace
        self.current_trace["epoch"].update(self.mrr_job.current_trace['epoch'])
        self.current_trace["epoch"].update(
            dict(
                epoch_time=epoch_time,
                avg_f1=mean_f1,
                event="eval_completed",
            )
        )
        self.mrr_job.current_trace["epoch"] = None