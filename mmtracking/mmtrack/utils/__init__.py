<<<<<<< HEAD
# Copyright (c) OpenMMLab. All rights reserved.
from .collect_env import collect_env
from .logger import get_root_logger

__all__ = ['collect_env', 'get_root_logger']
=======
# Copyright (c) OpenMMLab. All rights reserved.
from .collect_env import collect_env
from .logger import get_root_logger
from .util_distribution import build_ddp, build_dp, get_device

__all__ = [
    'collect_env', 'get_root_logger', 'build_ddp', 'build_dp', 'get_device'
]
>>>>>>> e79491ec8f0b8c86fda947fbaaa824c66ab2a991
