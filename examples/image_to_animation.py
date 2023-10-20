# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from image_to_annotations1 import image_to_annotations
from annotations_to_animation import annotations_to_animation
from pathlib import Path
import logging
import sys
from pkg_resources import resource_filename
from utils import crop, mask

def image_to_animation(img_fn: str, char_anno_dir: str, motion_cfg_fn: str, retarget_cfg_fn: str):
    """
    Given the image located at img_fn, create annotation files needed for animation.
    Then create animation from those animations and motion cfg and retarget cfg.
    """
    # create the annotations
    image_to_annotations(img_fn, char_anno_dir)
    
    crop(char_anno_dir)
    mask(char_anno_dir)
    
    """
    여기에 사진 수정하는 함수 넣어주면 될 듯 해요
    crop(char_anno_dir)
    mask(char_anno_dir)
    joint(char_anno_dir)
    """

    """
    여기에 사진 수정하는 함수 넣어주면 될 듯 해요
    crop(char_anno_dir)
    mask(char_anno_dir)
    joint(char_anno_dir)
    """

    # create the animation
    annotations_to_animation(char_anno_dir, motion_cfg_fn, retarget_cfg_fn)


if __name__ == '__main__':
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.DEBUG)

    img_fn = sys.argv[1]
    char_anno_dir = sys.argv[2]
    if len(sys.argv) > 3:
        motion_cfg_fn = sys.argv[3]
    else:
        motion_cfg_fn = resource_filename(__name__, 'config/motion/wave_hello.yaml')
    if len(sys.argv) > 4:
        retarget_cfg_fn = sys.argv[4]
    else:
        retarget_cfg_fn = resource_filename(__name__, 'config/retarget/fair1_ppf.yaml')

    image_to_animation(img_fn, char_anno_dir, motion_cfg_fn, retarget_cfg_fn)

def ani_main(_img_fn: str, _char_anno_dir: str):
    img_fn = _img_fn
    char_anno_dir = _char_anno_dir
    motion_cfg_fn = resource_filename(__name__, 'config/motion/hi.yaml')
    retarget_cfg_fn = resource_filename(__name__, 'config/retarget/cmu1_pfp_copy.yaml')
    image_to_animation(img_fn, char_anno_dir, motion_cfg_fn, retarget_cfg_fn)