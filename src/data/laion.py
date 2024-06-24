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

def safe_image_open(x):
    try:
        return Image.open(io.BytesIO(x[1].read())).convert('RGB')
    except IOError as e:
        print(f"Failed to open image: {e}")
        return None  # or use a placeholder image


# add validation checking
def build_laion_tar_images_datapipelines(images_tar_dir, image_transform, batch_size=None, *args, **kwargs):
    if isinstance(images_tar_dir, str):
        data_dir = list(braceexpand(images_tar_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=True)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    # datapipe = datapipe.load_from_tar()
    datapipe = TarArchiveLoaderWoException(datapipe)
    # filter out non-image content
    datapipe = datapipe.filter(lambda x: x[0].endswith('.jpg') or x[0].endswith('.jpeg') or x[0].endswith('.png'))
    # convert image content to PIL.Image
    # check valid image filter OSError: broken data stream when reading image file
    # datapipe = datapipe.map(lambda x: Image.open(io.BytesIO(x[1].read())).convert('RGB'))
    datapipe = datapipe.map(safe_image_open)
    # filter None image
    datapipe = datapipe.filter(lambda x: x is not None)
    # image_transform
    datapipe = datapipe.map(lambda x: image_transform(x))
    # add 'images' key to the data tuple
    datapipe = datapipe.map(lambda x: {'images': x})
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe


def build_random_image_datapipe(images_tar_dir, image_transform, batch_size=None, *args, **kwargs):
    import torchvision
    fake_dataset = torchvision.datasets.FakeData(size = 10000, image_size=(3, 1024, 1024), num_classes=1000)
    datapipe = dp.iter.IterableWrapper(fake_dataset)
    datapipe = datapipe.map(lambda x: image_transform(x[0]))
    datapipe = datapipe.map(lambda x: {'images': x})
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe
    