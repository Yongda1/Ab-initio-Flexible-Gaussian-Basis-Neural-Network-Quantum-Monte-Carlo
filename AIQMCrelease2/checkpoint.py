"""This module is for saving and reading."""
import dataclasses
import datetime
import os
from typing import Optional
import zipfile

from absl import logging
from AIQMCrelease2.wavefunction import nn
import jax
import jax.numpy as jnp
import numpy as np


def find_last_checkpoint(ckpt_path: Optional[str] = None) -> Optional[str]:
    if ckpt_path and os.path.exists(ckpt_path):
        files = [f for f in os.listdir(ckpt_path) if 'qmcjax_ckpt' in f]
        for file in sorted(files, reverse=True):
            fname = os.path.join(ckpt_path, file)
            with open(fname, 'rb') as f:
                try:
                    np.load(f, allow_pickle=True)
                    return fname
                except(OSError, EOFError, zipfile.BadZipFile):
                    logging.info('Error loading checkpoint %s. Trying next checkpoint...', fname)
    return None


def create_save_path(save_path: Optional[str]) -> str:
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
    default_save_path = os.path.join(os.getcwd(), f'AInet_{timestamp}')
    ckpt_save_path = save_path or default_save_path
    if ckpt_save_path and not os.path.isdir(ckpt_save_path):
        os.makedirs(ckpt_save_path)
    return ckpt_save_path


def get_restore_path(restore_path: Optional[str] = None) -> Optional[str]:
    if restore_path:
        ckpt_restore_path = restore_path
    else:
        ckpt_restore_path = None
    return ckpt_restore_path


def save(save_path: str,
         t: int,
         data: nn.AINetData,
         params,
         opt_state,) -> str:
    ckpt_filename = os.path.join(save_path, f'qmcjax_ckpt_{t:06d}.npz')
    logging.info('Saving checkpoint %s', ckpt_filename)
    with open(ckpt_filename, 'wb') as f:
        np.savez(
            f,
            t=t,
            data=dataclasses.asdict(data),
            params=params,
            opt_state=opt_state,
        )
    return ckpt_filename


def restore(restore_filename: str, batch_size: Optional[int] = None):
    logging.info('Loading checkpoint %s', restore_filename)
    with open(restore_filename, 'rb') as f:
        ckpt_data = np.load(f, allow_pickle=True)
        t = ckpt_data['t'].tolist() + 1
        data = nn.AINetData(**ckpt_data['data'].item())
        params = ckpt_data['params'].tolist
        opt_state = ckpt_data['opt_state'].tolist()
    return t, data, params, opt_state