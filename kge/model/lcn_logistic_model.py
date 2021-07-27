import torch
from torch import Tensor
from kge import Config, Dataset
from kge.model.kge_model import KgeModel
import json
import os
import numpy as np
import time

class LCNLogisticModel(KgeModel):
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


        types_path = self.config.get('lcn_logistic_model.types_path')
        y, idx, pos_weights, hier_tuple_ids, hier = self.load_types(types_dataset_path=types_path,
                                                                    num_entities=dataset.num_entities())
        self.types = y
        self.type_ids = idx
        self.hier_tuple_ids = hier_tuple_ids
        self.hier = hier
        self.pos_weights = pos_weights

        #predictions
        self.beta = self.config.get("lcn_logistic_model.beta")
        # penalties
        self.local_lamb = self.config.get("lcn_logistic_model.local_lambda")
        self.global_lamb = self.config.get("lcn_logistic_model.global_lambda")
        self.p = self.config.get("lcn_logistic_model.p")

        # materialize local training mask to avoid repetitive computation
        # lags y_train by one parent level for local training, repects multiple parents
        self.train_mask = self.__build_mask(y=y[idx['train']])

        # local predictors trained locally
        self.local = torch.nn.Linear(in_features=self._embedding_model.get_s_embedder().dim,
                                     out_features=y.size(1), bias=True)

        # global predictors receiving all data as well as all local scores
        self.glob = torch.nn.Linear(in_features=self._embedding_model.get_s_embedder().dim+y.size(1),
                                    out_features=y.size(1), bias=True)

    def prepare_job(self, job, **kwargs):
        self._embedding_model.prepare_job(job, **kwargs)

    def penalty(self, **kwargs):
        ''' penelize logistic model '''
        penalties = self._embedding_model.penalty(**kwargs)
        if "batch" in kwargs and "types" in kwargs["batch"]:

            # penalize local
            params = torch.cat([x.view(-1) for x in self.local.parameters()])
            penalties += [
                (
                    f"types_logistics.L{self.p}_penalty",
                    (self.local_lamb * torch.norm(params, p=self.p)).sum() / len(kwargs["batch"]["idx"]),
                )
            ]

            # penalize global
            params = torch.cat([x.view(-1) for x in self.glob.parameters()])
            penalties += [
                (
                    f"types_logistics.L{self.p}_penalty",
                    (self.global_lamb * torch.norm(params, p=self.p)).sum() / len(kwargs["batch"]["idx"]),
                )
            ]
        return penalties

    def get_tuple_ids(self):
        return self.hier_tuple_ids

    def get_train_mask(self, idx):
        return self.train_mask[idx]

    def get_s_embedder(self):
        return self._embedding_model.get_s_embedder()

    def get_o_embedder(self):
        return self._embedding_model.get_o_embedder()

    def get_p_embedder(self):
        return self._embedding_model.get_p_embedder()

    def get_scorer(self):
        return self._embedding_model.get_scorer()

    def get_local(self):
        return self.local

    def get_global(self):
        return self.glob

    def score_spo(self, s, p, o, direction=None):
        return self._embedding_model.score_spo(s, p, o, direction)

    def score_po(self, p, o, s=None):
        return self._embedding_model.score_po(p, o, s)

    def score_so(self, s, o, p=None):
        return self._embedding_model.score_so(s, o, p)

    def score_sp_po(self, s, p, o, entity_subset=None):
        return self._embedding_model.score_sp_po(s, p, o, entity_subset)

    # There are two ways to call the hierarchical training procedure:
    #   Supervised: Used for training, only passes entities with assigned parent, handeled in training job
    #   Unsupervised: Passes only entities with predicted parents to children, handeled by model itself

    # Utility methods
    def score_type(self, type_tuple, X):
        # returns extra dimension
        self.to_penalize[type_tuple] = len(X)
        return torch.squeeze(self.lcn_logistic[type_tuple](X), 2)

    def score_all(self, X):
        return torch.stack([torch.squeeze(linear(X), 1) for _, linear in self.lcn_logistic.items()]).transpose(0, 1)
    '''
    def score_type(self, type_tuple, idx):
        X = self._embedding_model.get_s_embedder().embed(indexes=idx)
        return self.lcn_logistic[type_tuple](X)

    def predict_proba(self, type_tuple, idx):
        return torch.sigmoid(self.score_type(type_tuple, idx))

    def predict(self, type_tuple, idx, thresh):
        y_proba = self.predict_proba(type_tuple=type_tuple, idx=idx)
        y_proba = y_proba.cpu().numpy()
        thresh = thresh.cpu().numpy()
        y_hat = (y_proba > thresh).astype(np.int)
        return y_hat
    '''
    def predict_all(self, idx, thresh, device):
        thresh = thresh.to(device)
        self.local.to(device)
        self.glob.to(device)
        X = self._embedding_model.get_s_embedder().to(device).embed(indexes=idx)
        local_conf = torch.sigmoid(self.local(X))
        y_hat_local = torch.where(local_conf > thresh, 1, 0)

        # mask confidences where parent was not predicted
        local_mask = self.__build_mask(y_hat_local, mode='valid', device=device)
        local_conf = local_conf * local_mask
        global_conf = torch.sigmoid(self.glob(torch.cat((X, local_conf), dim=1)))

        model_conf = self.beta*local_conf + (1 - self.beta)*global_conf
        y_hat = torch.where(model_conf > thresh, 1, 0)

        # mask inconsistent predictions - should be penalized in loss
        y_hat_mask = self.__build_mask(y_hat, mode='valid', device=device)
        return y_hat*y_hat_mask

    # function to zero all scores, that dont have the relative parent type assigned
    # used for local predictors
    #   corresponds to the siblings negative sampling strategy
    #   or prediction for child types, where a parent type was assigned
    def __build_mask(self, y, mode='train', device=None):
        if mode == 'train':
            # build tensor with respect to total size
            y_idx = self.type_ids['train']
            mask = np.zeros((self.types.size(0), self.types.size(1)))
            y = y.numpy()
            y_parent = {}
            # Assume root type is given for all instances
            for root_type in self.hier[0].keys():
                y_parent[(0, root_type)] = np.ones(len(y), dtype=np.int)

            for hier_tuple, tuple_id in self.hier_tuple_ids.items():
                mask[y_idx, tuple_id] = y_parent[hier_tuple]
                type_level, type_id = hier_tuple
                for child in self.hier[type_level][type_id]:
                    child_tuple = (type_level + 1, child)
                    if child_tuple not in y_parent:
                        y_parent[child_tuple] = y[:, tuple_id]
                    else:
                        # if other parent not predictied, overwrite decision
                        y_parent[child_tuple] = np.logical_or(y_parent[child_tuple], y[:, tuple_id]).astype(np.int)
            return torch.from_numpy(mask.astype(np.float32))

        mask = []
        y_parent = {}
        #Assume root type is given for all instances
        for root_type in self.hier[0].keys():
            y_parent[(0, root_type)] = torch.ones(len(y), dtype=torch.int).to(device)

        for hier_tuple, tuple_id in self.hier_tuple_ids.items():
            mask.append(y_parent[hier_tuple])
            type_level, type_id = hier_tuple
            for child in self.hier[type_level][type_id]:
                child_tuple = (type_level + 1, child)
                if child_tuple not in y_parent:
                    y_parent[child_tuple] = y[:, tuple_id]
                else:
                    # if other parent not predictied, overwrite decision
                    y_parent[child_tuple] = torch.logical_or(y_parent[child_tuple], y[:, tuple_id]).int()

        return torch.stack(mask).transpose(0, 1).float()

    def load_types(self, types_dataset_path, num_entities):

        # load the hierarchy to receive type information
        hier_path = os.path.join(types_dataset_path, 'hier.json')
        with open(hier_path, 'r') as json_file:
            hier = json.load(json_file)

        # reshape hierarchy and build binary type map
        hier_t, train_freq, hier_tuple_ids = {}, [], {}
        for hier_level, parents in hier.items():
            hier_t[int(hier_level)] = {}
            for level_type in parents:
                hier_t[int(hier_level)][int(level_type)] = hier[hier_level][level_type].copy()
                if (int(hier_level), int(level_type)) not in hier_tuple_ids:
                    hier_tuple_ids[(int(hier_level), int(level_type))] = len(hier_tuple_ids)
                    train_freq.append(0)
                    # fuse hierarchy keys to avoid nested dictionary
        hier = hier_t

        # build type maps keeping track of ids of respective split
        type_idx = {}
        y = np.zeros((num_entities, len(hier_tuple_ids)))
        # load types
        for split in ['train', 'valid', 'test']:
            idx = []
            types_path = os.path.join(types_dataset_path, split + '.del')
            with open(types_path, 'r') as file:
                for line in file:
                    entity_id, type_list = line.split("\t", maxsplit=1)
                    type_list = type_list.rstrip("\n")
                    # iterate through hierarchichal type structure
                    for level, level_types in enumerate(json.loads(type_list)):
                        for type_id in level_types:
                            bin_type_id = hier_tuple_ids[(level, int(type_id))]
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
        return y, idx, pos_weights, hier_tuple_ids, hier
