import argparse
import configparser
import cv2
import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN, FSRCNN, SRCNN_WO_1, SRCNN_WO_2, FSRCNN_S1, FSRCNN_S2

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', type=str, required=True)
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config_file)

config = configparser.ConfigParser()
config.read('testConfigs/' + args.config_file)

model_file 		= config['TEST']['ModelFile']
image_file 		= config['TEST']['ImageFile']
output_dir 	    = config['TEST']['OutputDir']
scale           = int(config['TEST']['Scale'])
model           = config['TEST']['Model']

output_dir = output_dir + "/" + model + "_X" + str(scale)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SRCNN_related_model = ['SRCNN', 'SRCNN_WO_1', 'SRCNN_WO_2']

if model == 'SRCNN':
    model = SRCNN().to(device)
elif model == 'FSRCNN':
    model = FSRCNN(scale_factor=scale).to(device)
elif model == 'SRCNN_WO_1':
    model = SRCNN_WO_1().to(device)
elif model == 'SRCNN_WO_2':
    model = SRCNN_WO_2().to(device)
elif model == 'FSRCNN_S1':
    model = FSRCNN_S1(scale_factor=scale).to(device)
elif model == 'FSRCNN_S2':
    model = FSRCNN_S2(scale_factor=scale).to(device)

state_dict = model.state_dict()
for n, p in torch.load(model_file, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        state_dict[n].copy_(p)
    else:
        raise KeyError(n)

model.eval()

image = pil_image.open(image_file).convert('RGB')

image_width = (image.width // scale) * scale
image_height = (image.height // scale) * scale

if model in SRCNN_related_model:
    image = image.resize((image_width, image_height), resample = pil_image.BICUBIC)
    image = image.resize((image.width // scale, image.height // scale), resample = pil_image.BICUBIC)
    image = image.resize((image.width * scale, image.height * scale), resample = pil_image.BICUBIC)
    image.save(image_file.replace('.', '_bicubic_x{}.'.format(scale)))

    image = np.array(image).astype(np.float32)
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))
else:
    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // scale, hr.height // scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
    bicubic.save(image_file.replace('.', '_bicubic_x{}.'.format(scale)))

    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)

    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)

    psnr = calc_psnr(hr, preds)
    print('PSNR: {:.2f}'.format(psnr))

preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
output = np.clip(cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR), 0.0, 255.0).astype(np.uint8)
output = pil_image.fromarray(output)
output.save(image_file.replace('.', '_srcnn_x{}.'.format(scale)))