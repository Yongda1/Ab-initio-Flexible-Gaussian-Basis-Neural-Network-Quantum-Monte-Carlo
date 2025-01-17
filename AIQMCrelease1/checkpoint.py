"""This module is for saving and reading."""
import dataclasses
import datetime
import os
from typing import Optional
import zipfile

from absl import logging
from AIQMCrelease1.wavefunction import nn
import jax
import jax.numpy as jnp
import numpy as np

