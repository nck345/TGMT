"""
Official Code Implementation of:
"D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping"
"""

import argparse
import time

import cv2
import glob
import numpy as np
import os
from skimage import measure

from loader.dataset_doc3d_grid_HV import gradient

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch
import torch.nn.functional as F
from networks.d2dewarp_model import D2DewarpModel

from PIL import Image
from torch import nn


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', default=448, help='image size')
    parser.add_argument('--model_path',
                        default='./model_weight.pt',
                        help='model path')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_img_path = os.path.join(os.path.dirname(base_dir), 'DocTr-Plus', 'data benchmark', 'crop', 'crop')
    parser.add_argument('--img_path',
                        default=default_img_path if os.path.exists(default_img_path) else './demo',
                        help='image path or path to folder containing images (set multi as true)')
    parser.add_argument('--save_path', default='./infer/DocUNet', help='save path')

    parser.add_argument('--plus', type=str2bool, default=True,
                        help='Enable D2Dewarp+ enhancements: adaptive white-balancing and contour boundary cropping')

    """ Model """
    parser.add_argument('--hv_out_chans', type=int, default=1, help='h line and v line output channels')
    parser.add_argument('--d_model', type=int, default=448, help='last layer dim in UNet and all layers in attention')
    parser.add_argument('--in_chans', type=int, default=4, help='input channels')
    return parser.parse_args()


parser = get_args()

recti_model = D2DewarpModel(img_size=parser.input_size, in_chans=parser.in_chans, hv_out_chans=parser.hv_out_chans,
                            d_model=parser.d_model)

if torch.cuda.is_available():
    recti_model = torch.nn.DataParallel(recti_model).cuda()
    state_dict = torch.load(parser.model_path, weights_only=False)
    recti_model.load_state_dict(state_dict['state_dict'])
else:
    state_dict = torch.load(parser.model_path, map_location='cpu', weights_only=False)
    sd = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
    new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    recti_model.load_state_dict(new_sd)
    recti_model = recti_model.to(device)
print(f'model loaded')


def predict(img_path, save_path, filename):
    assert os.path.exists(img_path), 'Incorrect Image Path'
    assert os.path.exists(save_path), 'Incorrect Save Path'

    dewarp_path = os.path.join(save_path, 'dewarp/')
    dewarp_plus_path = os.path.join(save_path, 'dewarp_plus/')
    pred_h_path = os.path.join(save_path, 'pred_h/')
    pred_v_path = os.path.join(save_path, 'pred_v/')
    if not os.path.exists(dewarp_path):
        os.makedirs(dewarp_path)
    if parser.plus and not os.path.exists(dewarp_plus_path):
        os.makedirs(dewarp_plus_path)
    if not os.path.exists(pred_h_path):
        os.makedirs(pred_h_path)
    if not os.path.exists(pred_v_path):
        os.makedirs(pred_v_path)

    img_size = parser.input_size
    img = np.array(Image.open(img_path).convert("RGB"))

    img_h, img_w, _ = img.shape
    input = cv2.resize(img, (img_size, img_size)) / 255.

    edge = gradient(img)
    edge = edge[:, :, np.newaxis]
    edge = cv2.resize(edge.astype(np.uint8), (img_size, img_size)) / 255.

    recti_model.eval()

    with torch.no_grad():
        input_ = torch.from_numpy(input).permute(2, 0, 1).to(device)
        input_ = input_.unsqueeze(0)
        edge = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).to(device)
        start = time.time()
        pred_h_lst, pred_v_lst, pred_bm = recti_model(torch.cat((input_.float(), edge.float()), dim=1))
        bm = (2. * (pred_bm / (parser.input_size)) - 1) * 1.004
        ps_time = time.time() - start
    # ----------------------------------------------------
    print("-----bm-----")
    print(bm)
    bm = bm.cpu()
    bm0 = cv2.resize(bm[0, 0].numpy(), (img_w, img_h))  # x flow
    bm1 = cv2.resize(bm[0, 1].numpy(), (img_w, img_h))  # y flow

    bm0 = cv2.blur(bm0, (3, 3))
    bm1 = cv2.blur(bm1, (3, 3))
    lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)  # h * w * 2

    out = F.grid_sample(torch.from_numpy(img / 255.).permute(2, 0, 1).unsqueeze(0).float().to(device), lbl.to(device),
                        align_corners=True)
    img_geo = ((out[0] * 255).permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
    save_filename = os.path.basename(filename)
    bgr_geo = img_geo[:, :, ::-1]
    cv2.imwrite(os.path.join(dewarp_path, save_filename), bgr_geo)  # save baseline

    if parser.plus:
        try:
            from model_utils.d2dewarp_plus import apply_d2dewarp_plus
            img_plus = apply_d2dewarp_plus(bgr_geo)
            cv2.imwrite(os.path.join(dewarp_plus_path, save_filename), img_plus)  # save D2Dewarp+
        except Exception as e:
            print(f"D2Dewarp+ postprocessing error: {e}")

    # -----------------------------------------------------
    pred_h = pred_h_lst[-1].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    pred_h = (pred_h > 0.2)
    pred_h = (pred_h * 255.).astype(np.uint8)
    cv2.imwrite(os.path.join(pred_h_path, save_filename),
                cv2.resize(pred_h, (img_w, img_h)))  # save

    pred_v = pred_v_lst[-1].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    pred_v = (pred_v > 0.2)
    pred_v = (pred_v * 255.).astype(np.uint8)
    cv2.imwrite(os.path.join(pred_v_path, save_filename),
                cv2.resize(pred_v, (img_w, img_h)))  # save

    return ps_time


if __name__ == '__main__':
    img_path = parser.img_path
    save_path = parser.save_path
    total_time = 0.0

    start = time.time()
    img_num = 0.0

    for file in glob.glob(os.path.join(img_path, "*")):
        if not os.path.isfile(file):
            continue
        print("file: ", file)
        base_name = os.path.splitext(os.path.basename(file))[0] + ".png"
        filename = os.path.join(save_path, base_name)

        total_time += predict(file, save_path, filename)
        print("total_time: ", total_time)
        img_num += 1
    if img_num > 0:
        print('FPS: %.1f' % (1.0 / (total_time / img_num)))

