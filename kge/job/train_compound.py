import torch
import torch.utils.data
import math

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn


class TrainingJobCompound(TrainingJob):
    """
    Combines n base trainers into a single training job. Each batch is given to
    the corresponding base trainer. The total number of batches in this trainer
    are the sum of the set of batches of each base trainer.
    Each base trainer has independent training settings, e.g. loss function.
    """

    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        config.log("Initializing compound training job...")
        self.type_str = "compound"
        self._base_trainer_iterators = {}
        self._normalize_weights = config.get("compound.normalize_weights")

        if self.__class__ == TrainingJobCompound:
            for f in Job.job_created_hooks:
                f(self)

        # fix batch size to 1
        # ensures a single trainer per batch, because this dataloader samples
        # batches, not examples
        self.batch_size = 1

        # initialize base trainers
        self._base_trainers = {}
        total_weight = 0
        for trainer_id, trainer in enumerate(config.get("compound.trainers")):
            # create config for base trainer
            base_config = config.clone()

            # overwrite train entries for base trainer
            trainer_key = "compound.trainers.{}".format(trainer)
            for entry, value in config.get(trainer_key).items():
                if entry == "loss_weight" and self._normalize_weights:
                    total_weight += value
                    continue
                key = "train.{}".format(entry)
                base_config.set(key, value)

            # init base trainer
            self._base_trainers[trainer_id] = self.create(
                base_config,
                dataset,
                self,
                model=self.model,
                forward_only=forward_only
            )

        # normalize weights
        if self._normalize_weights:
            for trainer in self._base_trainers.values():
                trainer._loss_weight = trainer._loss_weight / total_weight

    def _prepare(self):
        super()._prepare()

        # prepare base trainers
        self.num_examples = 0
        trainer_datasets = []
        for trainer_id in self._base_trainers:
            trainer = self._base_trainers[trainer_id]
            trainer._prepare()
            trainer_size = math.ceil(
                len(trainer.loader.dataset) / trainer.batch_size
            )
            self.num_examples += len(
                trainer.loader.dataset
            )
            trainer_datasets.append(
                torch.ones(trainer_size, dtype=torch.long) * trainer_id
            )
        main_dataset = torch.cat(trainer_datasets)

        # prepare this trainer
        self.loader = torch.utils.data.DataLoader(
            dataset=main_dataset.tolist(),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _get_collate_fun(self):
        def collate(batch):
            """
            Calls collate function of the base trainer that corresponds
            to the given batch.
            Returns object of the form (trainer_name, base_batch) where
            base_batch is the object returned by the base trainer's collate
            function.
            """

            trainer_id = batch[0]
            # get batch from relevant base trainer
            # already goes through collate function
            base_batch = next(self._base_trainer_iterators[trainer_id])
            if self._base_trainers[trainer_id].type_str == "HCMN" :
                return {"types": base_batch["types"],
                        "idx": base_batch["idx"],
                        "processed_batch": (trainer_id, base_batch)}
            else:
                return {"triples": base_batch["triples"],
                        "processed_batch": (trainer_id, base_batch)}

        return collate

    def _prepare_batch(
            self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        pass

    def _process_batch(
            self,
            batch_index,
            batch,
    ):
        trainer_id = batch["processed_batch"][0]
        processed_batch = batch["processed_batch"][1]

        return self._base_trainers[trainer_id]._process_batch(
            batch_index,
            processed_batch
        )

    def run_epoch(self):
        # create base trainer iterators for current epoch
        for trainer_id, trainer in self._base_trainers.items():
            self._base_trainer_iterators[trainer_id] = iter(trainer.loader)

        return super().run_epoch()
