import cv2 as cv
import numpy as np
import torch
from SuperGlue.models.matching import Matching
from SuperGlue.models.utils import process_resize
import datetime


def read_image(image, p_device, resize, resize_float):
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    if resize_float:
        image = cv.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv.resize(image, (w_new, h_new)).astype('float32')
    inp = torch.from_numpy(image / 255.).float()[None, None].to(p_device)
    return inp


def superpoint_superglue(img_1, img_2):
    torch.set_grad_enabled(False)
    # Load the SuperPoint and SuperGlue models.
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',  # outdoor
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    start = datetime.datetime.now()
    matching = Matching(config).eval().to(device)
    # Load the image pair.
    # inp0 = read_image(cv.cvtColor(img_1, cv.COLOR_BGR2GRAY), device, [640, 480], True)
    # inp1 = read_image(cv.cvtColor(img_2, cv.COLOR_BGR2GRAY), device, [640, 480], True)
    inp0 = read_image(cv.cvtColor(img_1, cv.COLOR_BGR2GRAY), device, [int(img_1.shape[1]), int(img_1.shape[0])], True)
    inp1 = read_image(cv.cvtColor(img_2, cv.COLOR_BGR2GRAY), device, [int(img_2.shape[1]), int(img_2.shape[0])], True)
    # Perform the matching.
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    valid = matches != -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    mkpts_1_np = np.array(mkpts0)
    mkpts_2_np = np.array(mkpts1)
    mconf_np = np.array(mconf)
    inds = mconf_np.argsort()[::-1]
    sorted_mkpts_1 = mkpts_1_np[inds]
    sorted_mkpts_2 = mkpts_2_np[inds]
    matched_kpts = []
    for m_id, matches in enumerate(sorted_mkpts_1):
        matched_kpts.append((matches, sorted_mkpts_2[m_id]))
    duration = datetime.datetime.now() - start

    return sorted_mkpts_1, sorted_mkpts_2, matched_kpts, duration
