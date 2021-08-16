import torch
from torch import Tensor
from kge import Config, Dataset
from kge.model.kge_model import KgeModel
import json
import os
import numpy as np
import time

class hmcn_model(KgeModel):

    """
        Implements hierarchical Multi-Label classification Network as defined in Wehrmann et al. (2018)
        Codes are adapted from:
        https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/master/model/classification/hmcn.py

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


        types_path = self.config.get('hmcn_model.types_path')
        y, idx, pos_weights, hier_tuple_ids, hier, hierarchical_depth, global2local, hierarchy_classes\
            = self.load_types(types_dataset_path=types_path, num_entities=dataset.num_entities())
        self.types = y
        self.type_ids = idx
        self.hier_tuple_ids = hier_tuple_ids
        self.hier = hier
        self.pos_weights = pos_weights

        #HMCN setup
        self.hierarchical_depth = hierarchical_depth
        self.hierarchical_class = hierarchy_classes
        self.global2local = global2local
        hidden_dimension = self._embedding_model.get_s_embedder().dim

        #predictions
        self.beta = self.config.get("hmcn_model.beta")
        self.p = self.config.get("hmcn_model.hiddenlayer_dropout")
        self.lamb = torch.Tensor([self.config.get("hmcn_model.lambda")])

        # Setup HMCN model according to Wehrmann et al. (2018)
        # Code adapted from
        # https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/master/model/classification/hmcn.py
        self.local_layers = torch.nn.ModuleList()
        self.global_layers = torch.nn.ModuleList()
        for i in range(1, len(self.hierarchical_depth)):
            self.global_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dimension + self.hierarchical_depth[i - 1], self.hierarchical_depth[i]),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(self.hierarchical_depth[i]),
                    torch.nn.Dropout(p=0.5)
                ))
            self.local_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.hierarchical_depth[i], self.global2local[i]),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(self.global2local[i]),
                    torch.nn.Linear(self.global2local[i], self.hierarchical_class[i])
                ))

        self.global_layers.apply(self._init_weight)
        self.local_layers.apply(self._init_weight)
        self.linear = torch.nn.Linear(self.hierarchical_depth[-1], len(hier_tuple_ids))
        self.linear.apply(self._init_weight)
        self.dropout = torch.nn.Dropout(p=self.p)

    def prepare_job(self, job, **kwargs):
        self._embedding_model.prepare_job(job, **kwargs)

    def penalty(self, **kwargs):
        ''' penalty calculated in training as it depends on confidence estimates '''
        penalties = self._embedding_model.penalty(**kwargs)
        return penalties

    def get_lambda(self):
        return self.lamb

    def get_tuple_ids(self):
        return self.hier_tuple_ids

    def get_train_mask(self, idx):
        return self.train_mask[idx]

    # pass embedding methods down to wrapped embedder
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

    # mimics forward
    def predict_all(self, idx, device):

        entity_embeddings = self._embedding_model.get_s_embedder().to(device).embed(indexes=idx)
        local_layer_outputs = []
        global_layer_activation = entity_embeddings
        #batch_size = len(idx)
        for i, (local_layer, global_layer) in enumerate(zip(self.local_layers, self.global_layers)):
            local_layer_activation = global_layer(global_layer_activation)
            local_layer_outputs.append(local_layer(local_layer_activation))
            if i < len(self.global_layers) - 1:
                global_layer_activation = torch.cat((local_layer_activation, entity_embeddings), 1)
            else:
                global_layer_activation = local_layer_activation

        global_layer_output = self.linear(global_layer_activation)
        local_layer_output = torch.cat(local_layer_outputs, 1)
        probits = self.beta * torch.sigmoid(local_layer_output) + (1 - self.beta) * torch.sigmoid(global_layer_output)
        return global_layer_output, local_layer_output, probits

    # function to zero all child class predictions, that dont have the relative parent type assigned
    # type = 'proba' used for violation calculation by lagging parent confidence to child
    # type = 'binbary' used to ensure hierarchy consistent prediction
    def build_mask(self, y, type='binary', device=None):

        mask = []
        y_parent = {}
        # Assume root type is predicted for all instances
        for root_type in self.hier[1].keys():
            if type == 'binary':
                y_parent[(1, root_type)] = torch.ones(len(y), dtype=torch.int).to(device)
            else:
                y_parent[(1, root_type)] = torch.ones(len(y), dtype=torch.float).to(device)

        for hier_tuple, tuple_id in self.hier_tuple_ids.items():
            mask.append(y_parent[hier_tuple])
            type_level, type_id = hier_tuple
            for child in self.hier[type_level][type_id]:
                child_tuple = (type_level + 1, child)
                if child_tuple not in y_parent:
                    y_parent[child_tuple] = y[:, tuple_id]
                # DAG!
                else:
                    if type == 'binary':
                        # Tie handling when both parent cast predictions: logical or
                        y_parent[child_tuple] = torch.logical_or(y_parent[child_tuple], y[:, tuple_id]).int()
                    else:
                        # Tie Handling use maximum confidence of parent predictions
                        y_parent[child_tuple] = torch.max(y_parent[child_tuple], y[:, tuple_id]).float()

        return torch.stack(mask).transpose(0, 1).float()

    def load_types(self, types_dataset_path, num_entities):
        """
        @param types_dataset_path: Path to type dataset. Requires hier.json, train.del, valid.del and test.del.
        @param num_entities: Number of unique entities in the KG.
        @return:
            y: Binary map of types with shape (num_entities, num_types-1). Root type not considered.
            idx: dict with keys ['train', 'valid', 'test'] containing respective entity ids.
            pos_weights: positive weights of class computed from training split for weighted bce_with_logits_loss.
            hier_tuple_ids: dict with keys [(level, type_id)] for mapping y to type id
            hierarchical_depth: number of ReLU neurons per hierarchy level: 384.
            global2local: local ReLU neurons.  same as hierarchy_classes.
            hierarchy_classes: Number of classes per hierarchy level. root class excluded.
        """
        # load the hierarchy to receive type information
        hier_path = os.path.join(types_dataset_path, 'hier.json')
        with open(hier_path, 'r') as json_file:
            hier = json.load(json_file)

        # reshape hierarchy and build binary type map (usefull to map predictions to type_ids)
        # build required shapes for HMCN
        # ReLU neurons set to 384 per level see Wehrmann et al (2018)
        hier_t, train_freq, hier_tuple_ids = {}, [], {}
        for hier_level, parents in hier.items():
            hier_t[int(hier_level)] = {}
            if int(hier_level) == 0:
                #no prediction for level 0
                hierarchical_depth = [0] # Global ReLU neurons
                global2local = [0] # Local transfer neurons
                hierarchy_classes = [0] #number of classes per level
                continue
            else:
                hierarchical_depth.append(384) # Global ReLU neurons
                global2local.append(len(parents)) # Local transfer neurons
                hierarchy_classes.append(len(parents)) # number of classes per level
            for level_type in parents:
                hier_t[int(hier_level)][int(level_type)] = hier[hier_level][level_type].copy()
                if (int(hier_level), int(level_type)) not in hier_tuple_ids:
                    hier_tuple_ids[(int(hier_level), int(level_type))] = len(hier_tuple_ids)
                    train_freq.append(0)
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
                    # iterate through hierarchical type structure
                    for level, level_types in enumerate(json.loads(type_list)):
                        if level == 0:
                            continue
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
        return y, idx, pos_weights, hier_tuple_ids, hier, hierarchical_depth, global2local, hierarchy_classes


    def _init_weight(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.1)