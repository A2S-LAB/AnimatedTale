import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import animated_drawings.render
from pathlib import Path
from datetime import datetime
import yaml
import shutil

counter = 0

def annotations_to_animation(char_anno_dir: str, motion_cfg_fn: str, retarget_cfg_fn: str):
    animated_drawing_dict = {
        'character_cfg': str(Path(char_anno_dir, 'char_cfg.yaml').resolve()),
        'motion_cfg': str(Path(motion_cfg_fn).resolve()),
        'retarget_cfg': str(Path(retarget_cfg_fn).resolve())
    }

    now = datetime.now()
    backup_file_name = now.strftime("%H%M%S")
    global counter
    counter = counter % 7 + 1

    # create mvc config
    mvc_cfg = {
        'scene': {'ANIMATED_CHARACTERS': [animated_drawing_dict]},  # add the character to the scene
        'controller': {
            'MODE': 'video_render',  # 'video_render' or 'interactive'
            'OUTPUT_VIDEO_PATH': str(Path(char_anno_dir, f'../web_contents/exhibit/video_{counter}.gif').resolve())},  # set the output location
        'view':{
            'USE_MESA': True}
    }

    # write the new mvc config file out
    output_mvc_cfn_fn = str(Path(char_anno_dir, 'mvc_cfg.yaml'))
    with open(output_mvc_cfn_fn, 'w') as f:
        yaml.dump(dict(mvc_cfg), f)

    # render the video
    animated_drawings.render.start(output_mvc_cfn_fn)
    shutil.copyfile(f'web_contents/exhibit/video_{counter}.gif', f'web_contents/exhibit_backup/video_{backup_file_name}.gif')
