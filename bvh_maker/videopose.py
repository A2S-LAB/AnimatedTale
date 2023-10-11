import os

from common.arguments import parse_args
from common.camera import *
from common.generators import UnchunkedGenerator
from common.loss import *
from common.model import *
from common.utils import evaluate, add_path
from numpy import *
import numpy as np
from bvh_skeleton import cmu_skeleton_copy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

metadata = {'layout_name': 'coco', 'num_joints': 17, 'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}
Cmu_skeleton = cmu_skeleton_copy.CMUSkeleton()

add_path()

class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 9, 10]


def load_model(args):
    model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=args.causal, dropout=args.dropout, channels=args.channels, dense=args.dense)
    if torch.cuda.is_available():
        print('Using Cuda...')
        model_pos = model_pos.cuda()

    # load trained model
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'])

    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2 
    causal_shift = 0
    return pad, causal_shift, model_pos

def render(args, pad, causal_shift, model_pos, aa, kpts):
    # npz = kpts
    keypoints = kpts  # (N, 17, 2)

    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=1000, h=1002)

    input_keypoints = keypoints.copy()
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                            pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, model_pos, return_predictions=True)
    rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
    prediction = camera_to_world(prediction, R=rot, t=0)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    prediction_copy = np.copy(prediction)

    for frame in prediction_copy:
        for point3d in frame:
            # point3d[0] *= 100
            # point3d[1] *= 100
            # point3d[2] *= 100

            X = point3d[0]
            Y = point3d[1]
            Z = point3d[2]

            point3d[0] = -X
            point3d[1] = Z
            point3d[2] = Y
            
    Cmu_skeleton.poses2bvh(prediction_copy, aa, output_file=f"./outputs/out_{aa}.bvh", )

def inference_video():
    args = parse_args()
    args.evaluate = 'pretrained_h36m_detectron_coco.bin'
    pad, causal_shift, model_pos = load_model(args)
    render(args, pad, causal_shift, model_pos)

def write_3d_point(prediction3dpoint):
    frameNum = 1
    for frame in prediction3dpoint:
        
        outfilename = os.path.join(f"./outputs/txt/out_3dpoint{frameNum}.txt")
        file = open(outfilename, 'w')
        frameNum += 1
        for point3d in frame:
            str = '{},{},{}\n'.format(point3d[0],point3d[1],point3d[2])
            file.write(str)
        file.close()
    
    print('3D points saved')
