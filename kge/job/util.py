import torch
from torch import Tensor
from typing import List, Union
import numpy as np
import json
from os import path

def get_sp_po_coords_from_spo_batch(
    batch: Union[Tensor, List[Tensor]], num_entities: int, sp_index: dict, po_index: dict
) -> torch.Tensor:
    """Given a set of triples , lookup matches for (s,p,?) and (?,p,o).

    Each row in batch holds an (s,p,o) triple. Returns the non-zero coordinates
    of a 2-way binary tensor with one row per triple and 2*num_entites columns.
    The first half of the columns correspond to hits for (s,p,?); the second
    half for (?,p,o).

    """
    if type(batch) is list:
        batch = torch.cat(batch).reshape((-1, 3)).int()
    sp_coords = sp_index.get_all(batch[:, [0, 1]])
    po_coords = po_index.get_all(batch[:, [1, 2]])
    po_coords[:, 1] += num_entities
    coords = torch.cat(
        (
            sp_coords,
            po_coords
        )
    )

    return coords


def coord_to_sparse_tensor(
    nrows: int, ncols: int, coords: Tensor, device: str, value=1.0, row_slice=None
):
    if row_slice is not None:
        if row_slice.step is not None:
            # just to be sure
            raise ValueError()

        coords = coords[
            (coords[:, 0] >= row_slice.start) & (coords[:, 0] < row_slice.stop), :
        ]
        coords[:, 0] -= row_slice.start
        nrows = row_slice.stop - row_slice.start

    if device == "cpu":
        labels = torch.sparse.FloatTensor(
            coords.long().t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
        )
    else:
        labels = torch.cuda.sparse.FloatTensor(
            coords.long().t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
            device=device,
        )

    return labels


def load_types( types_dataset_path, num_entities, split='train'):

    # load the hierarchy to receive type information
    hier_path = path.join(types_dataset_path, 'hier.json')
    with open(hier_path, 'r') as json_file:
        hier = json.load(json_file)

    bin_type_ids, type_freq = {}, []
    for hier_level in set(hier.keys()):
        for level_type in set(hier[hier_level].keys()):
            if int(level_type) not in bin_type_ids:
                bin_type_ids[int(level_type)] = len(bin_type_ids)
                type_freq.append(0)

    # load types
    idx = []
    y = np.zeros((num_entities, len(bin_type_ids)))
    if split == 'train':
        types_path = path.join(types_dataset_path, split+'.del')
    else:
        #only load train for frequencies
        types_path = path.join(types_dataset_path, 'train.del')
    with open(types_path, 'r') as file:
        for line in file:
            entity_id, type_list = line.split("\t", maxsplit=1)
            type_list = type_list.rstrip("\n")
            #iterate through hierarchichal type structure
            for level in json.loads(type_list):
                for type_id in level:
                    bin_type_id = bin_type_ids[int(type_id)]
                    y[int(entity_id), bin_type_id] = 1
                    type_freq[bin_type_id]+=1
            idx.append(int(entity_id))

    # compute weights for loss function
    pos_weights = []
    for class_count in type_freq:
        if class_count == 0:
            pos_weight = len(idx)
        else:
            neg_count = len(idx) - class_count
            pos_weight = neg_count / class_count
        pos_weights.append(pos_weight)

    #overwrite y and idx for valid and test
    if split != 'train':
        types_path = path.join(types_dataset_path, split + '.del')
        idx = []
        y = np.zeros((num_entities, len(bin_type_ids)))
        with open(types_path, 'r') as file:
            for line in file:
                entity_id, type_list = line.split("\t", maxsplit=1)
                type_list = type_list.rstrip("\n")
                # iterate through hierarchichal type structure
                for level in json.loads(type_list):
                    for type_id in level:
                        bin_type_id = bin_type_ids[int(type_id)]
                        y[int(entity_id), bin_type_id] = 1
                idx.append(int(entity_id))

    # create output numpy arrays and tensors
    y = torch.from_numpy(y)
    pos_weights = torch.from_numpy(np.array(pos_weights))
    idx = torch.from_numpy(np.array(idx))
    return y, pos_weights, idx
