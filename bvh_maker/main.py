import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
import sys

from utils import calculate_area
from SPPE.src.main_fast_inference import InferenNet_fast
from dataloader_webcam import WebcamLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from opt import opt

from videopose import load_model, render
from common.arguments import parse_args


args = opt
args.dataset = 'coco'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print("DEVICE = CUDA")

def loop():
    n = 0
    while True:
        yield n
        n += 1

if __name__ == "__main__":
    args_VP = parse_args()
    args_VP.evaluate = 'pretrained_h36m_detectron_coco.bin'
    pad, causal_shift, model_pos = load_model(args_VP)

    webcam = '0'
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    # Load input video
    data_loader = WebcamLoader(webcam).start()
    (fourcc, fps, frameSize) = data_loader.videoinfo()

    # Load detection loader
    print('Loading YOLO model..')
    sys.stdout.flush()
    args.detbatch = 128
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()
    det_processor = DetectionProcessor(det_loader).start()
    # Load pose model
    pose_dataset = Mscoco()
    pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    pose_model.to(device)
    pose_model.eval()

    # Data writer
    save_path = os.path.join(args.outputpath, 'AlphaPose_webcam' + webcam + '.avi')
    writer = DataWriter(args.vis, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()

    print('Starting webcam demo, press Ctrl + C to terminate...')
    sys.stdout.flush()
    im_names_desc = tqdm(loop())
    batchSize = args.posebatch

    aa = 0
    kpts = []  # 사람의 포즈 키포인트

    for i in im_names_desc: 
        try:
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                    continue

                # Pose Estimation
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)].to(device)
                    hm_j = pose_model(inps_j)
                    hm.append(hm_j)
                hm = torch.cat(hm)
                
                # hm = pose_model(inps.to(device))
                hm = hm.cpu().data
                writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])

            final_result = writer.results()
            kpts = []  # 사람의 포즈 키포인트
            no_person = []  # 감지된 사람이 없는 경우

            if len(final_result) >= 15:
                for i in range(len(final_result)):
                    if not final_result[i]['result']:  # No people
                        no_person.append(i)
                        kpts.append(None)
                        print('no Person')
                        continue

                    kpt = max(final_result[i]['result'],
                            key=lambda x: x['proposal_score'].data[0] * calculate_area(x['keypoints']), )['keypoints']

                    kpts.append(kpt.data.numpy())

                    for n in no_person:
                        kpts[n] = kpts[-1]
                    no_person.clear()

                writer.clear_results()

                kpts = np.array(kpts).astype(np.float32)
                render(args_VP, pad, causal_shift, model_pos, aa, kpts)
                aa += 1
                print(aa)
            
        except KeyboardInterrupt:
            break

    writer.stop()
