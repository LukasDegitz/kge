import torch
from torch import Tensor
from kge import Config, Dataset
from kge.model.kge_model import KgeModel
import json
import os
import numpy as np

class TypesLogisticModel(KgeModel):
    """Adds an hierarchical entity typing component to an embedding model.

    """

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)

        # Initialize embedding model
        embedding_model = KgeModel.create(
            config=config,
            dataset=dataset,
            configuration_key=self.configuration_key + ".embedding_model",
            init_for_load_only=init_for_load_only,
        )

        # Initialize this model
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=embedding_model.get_scorer(),
            create_embedders=False,
            init_for_load_only=init_for_load_only,
        )
        self._embedding_model = embedding_model


        types_path = self.config.get('types_logistic_model.types_path')
        y, loss_weights, idx = self.load_types(types_dataset_path=types_path, num_entities=dataset.num_entities())
        self.types = y
        self.type_ids = idx
        self.types_loss_weights = loss_weights
        self.linear = torch.nn.Linear(in_features=self._embedding_model.get_s_embedder().dim,
                                      out_features=y.size(1), bias=True)

    def prepare_job(self, job, **kwargs):
        self._embedding_model.prepare_job(job, **kwargs)

    def penalty(self, **kwargs):
        penalties = self._embedding_model.penalty(**kwargs)
        if "batch" in kwargs and "types" in kwargs["batch"]:
            params = torch.cat([x.view(-1) for x in self.linear.parameters()])
            lamb = self.config.get("types_logistic_model.lambda")
            p = self.config.get("types_logistic_model.p")
            penalties += [
                (
                    f"types_logistics.L2_penalty",
                    (lamb * torch.norm(params, p=p)).sum()/len(kwargs["batch"]["idx"]),
                )
            ]
        return penalties

    def get_linear(self):
        return self.linear

    def get_s_embedder(self):
        return self._embedding_model.get_s_embedder()

    def get_o_embedder(self):
        return self._embedding_model.get_o_embedder()

    def get_p_embedder(self):
        return self._embedding_model.get_p_embedder()

    def get_scorer(self):
        return self._embedding_model.get_scorer()

    def score_spo(self, s, p, o, direction=None):
        return self._embedding_model.score_spo(s, p, o, direction)

    def score_po(self, p, o, s=None):
        return self._embedding_model.score_po(p, o, s)

    def score_so(self, s, o, p=None):
        return self._embedding_model.score_so(s, o, p)

    def score_sp_po(self, s, p, o, entity_subset=None):
        return self._embedding_model.score_sp_po(s, p, o, entity_subset)

    def load_types(self, types_dataset_path, num_entities):

        # load the hierarchy to receive type information
        hier_path = os.path.join(types_dataset_path, 'hier.json')
        with open(hier_path, 'r') as json_file:
            hier = json.load(json_file)

        bin_type_ids, train_freq = {}, []
        for hier_level in set(hier.keys()):
            for level_type in set(hier[hier_level].keys()):
                if int(level_type) not in bin_type_ids:
                    bin_type_ids[int(level_type)] = len(bin_type_ids)
                    train_freq.append(0)

        type_idx = {}
        y = np.zeros((num_entities, len(bin_type_ids)))
        # load types
        for split in ['train', 'valid', 'test']:
            idx = []
            types_path = os.path.join(types_dataset_path, split + '.del')
            with open(types_path, 'r') as file:
                for line in file:
                    entity_id, type_list = line.split("\t", maxsplit=1)
                    type_list = type_list.rstrip("\n")
                    # iterate through hierarchichal type structure
                    for level in json.loads(type_list):
                        for type_id in level:
                            bin_type_id = bin_type_ids[int(type_id)]
                            y[int(entity_id), bin_type_id] = 1
                            if split == 'train':
                                train_freq[bin_type_id] += 1
                    idx.append(int(entity_id))
            type_idx[split] = idx.copy()

        # compute weights for loss function
        pos_weights = []
        for class_count in train_freq:
            if class_count == 0:
                pos_weight = len(type_idx['train'])
            else:
                neg_count = len(type_idx['train']) - class_count
                pos_weight = neg_count / class_count
            pos_weights.append(pos_weight)

        # create output numpy arrays and tensors
        y = torch.from_numpy(y)
        pos_weights = torch.from_numpy(np.array(pos_weights))
        idx = {split: torch.from_numpy(np.array(entity_ids)) for split, entity_ids in type_idx.items()}
        return y, pos_weights, idx
