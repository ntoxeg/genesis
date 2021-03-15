# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2020 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

import os

import torch
import torch.nn.functional as F

import numpy as np

import minerl

from forge import flags
from forge.experiment_tools import fprint

from utils.misc import loader_throughput


flags.DEFINE_integer('img_size', 64,
                     'Dimension of images. Images are square.')
flags.DEFINE_integer('val_frac', 60,
                     'Fraction of training images to use for validation.')

flags.DEFINE_integer('num_workers', 4, 'TF records dataset.')
flags.DEFINE_integer('buffer_size', 128, 'TF records dataset.')

flags.DEFINE_integer('K_steps', 7, 'Number of recurrent steps.')


SEED = 0


def load(cfg, **unused_kwargs):
    # Fix TensorFlow seed
    global SEED
    SEED = cfg.seed

    if cfg.num_workers == 0:
        fprint("Need to use at least one worker for loading.")
        cfg.num_workers = 1

    del unused_kwargs
    print(f"Using {cfg.num_workers} data workers.")
    # Create data iterators
    train_loader = MineRLLoader(
        mode='devel_train', img_size=cfg.img_size,
        val_frac=cfg.val_frac, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, buffer_size=cfg.buffer_size)
    val_loader = MineRLLoader(
        mode='devel_val', img_size=cfg.img_size,
        val_frac=cfg.val_frac, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, buffer_size=cfg.buffer_size)
    test_loader = MineRLLoader(
        mode='test', img_size=cfg.img_size,
        val_frac=cfg.val_frac, batch_size=1,
        num_workers=1, buffer_size=cfg.buffer_size)

    # Throughput stats
    loader_throughput(train_loader)

    return (train_loader, val_loader, test_loader)


def _make_iterator(reader, batch_size):
    """Make an iterator from a MineRL dataset"""
    for current_state, _, _, _, _ in reader.batch_iter(batch_size=batch_size, num_epochs=1, seq_len=1):
        yield current_state['pov'][:, 0, ...]

class MineRLLoader():
    """MineRL dataset"""

    def __init__(self, mode, img_size, val_frac, batch_size,
                 num_workers, buffer_size):
        self.data_folder = os.environ["MINERL_DATA_ROOT"]
        self.img_size = img_size
        self.batch_size = batch_size

        #  if "train" in mode:
            #  self.reader = minerl.data.make('MineRLNavigateDense-v0')
        #  elif "test" in mode:
            #  self.reader = minerl.data.make("MineRLObtainDiamond-v0")
        #  else:
            #  self.reader = minerl.data.make("MineRLTreechop-v0")

        self.reader = minerl.data.make("MineRLObtainDiamond-v0")

        self._iter = _make_iterator(self.reader, self.batch_size)
        # TODO: avoid hard coding these
        train_sz = 58541
        val_sz = 100000
        test_sz = 100000
        if mode == 'train':
            num_frames = train_sz
        elif mode == 'test':
            num_frames = test_sz
        elif mode == 'devel_train':
            # num_frames = int(train_sz * (1 - (val_frac / 100)))
            num_frames = train_sz
        elif mode == 'devel_val':
            # num_frames = int(train_sz * (val_frac / 100))
            num_frames = val_sz
        else:
            raise ValueError("Mode not known.")
        self.length = num_frames // batch_size

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        try:
            img = next(self._iter)
            img = np.moveaxis(img, 3, 1)
            img = torch.FloatTensor(img)
            if self.img_size != 64:
                img = F.interpolate(img, size=self.img_size)
            return {'input': img}
        except StopIteration:
            print("Reached end of epoch. Creating new iterator.")
            # Create new iterator for next epoch
            self._iter = _make_iterator(self.reader, self.batch_size)
            raise StopIteration
