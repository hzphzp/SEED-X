# -*- coding: utf-8 -*-
import torchdata.datapipes as dp
import json
from PIL import Image
import functools
import numpy as np
import torch
import torch.distributed as dist
import os
import random
from braceexpand import braceexpand
import hydra
from .datapipes import TarArchiveLoaderWoException
from .image_text_pairs_clm import base64_to_image
import io

# from .any_res import process_anyres_image, anyres_data_collate, anyres_data_collate_old

import pyrootutils
try:
    import fitz
except:
    fitz = None

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)


def build_laion_tar_images_datapipelines(images_tar_dir, *args, **kwargs):
    if isinstance(images_tar_dir, str):
        data_dir = list(braceexpand(images_tar_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=True)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    # datapipe = TarArchiveLoaderWoException(datapipe)
    # filter out non-image content
    datapipe = datapipe.filter(lambda x: x[0].endswith('.jpg') or x[0].endswith('.jpeg') or x[0].endswith('.png'))
    # convert image content to PIL.Image
    datapipe = datapipe.map(lambda x: Image.open(io.BytesIO(x[1].read())).convert('RGB'))
    return datapipe


