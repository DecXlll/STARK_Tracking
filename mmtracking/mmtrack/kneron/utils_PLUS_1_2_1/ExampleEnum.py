# ******************************************************************************
#  Copyright (c) 2021. Kneron Inc. All rights reserved.                        *
# ******************************************************************************


import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from enum import Enum

class ImageType(Enum):
    GENERAL = 'general'
    BINARY = 'binary'


class ImageFormat(Enum):
    RGB565 = 'RGB565'
    RGBA8888 = 'RGBA8888'
    YUYV = 'YUYV'
    RAW8 = 'RAW8'


class ResizeMode(Enum):
    NONE = 'none'
    ENABLE = 'auto'


class PaddingMode(Enum):
    NONE = 'none'
    PADDING_CORNER = 'corner'
    PADDING_SYMMETRIC = 'symmetric'


class PostprocessMode(Enum):
    NONE = 'none'
    YOLO_V3 = 'yolo_v3'
    YOLO_V5 = 'yolo_v5'


class NormalizeMode(Enum):
    NONE = 'none'
    KNERON = 'kneron'
    TENSORFLOW = 'tensorflow'
    YOLO = 'yolo'
    CUSTOMIZED_DEFAULT = 'customized_default'
    CUSTOMIZED_SUB128 = 'customized_sub128'
    CUSTOMIZED_DIV2 = 'customized_div2'
    CUSTOMIZED_SUB128_DIV2 = 'customized_sub128_div2'


class InferenceRetrieveNodeMode(Enum):
    FIXED = 'fixed'
    FLOAT = 'float'
