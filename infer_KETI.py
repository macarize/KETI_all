import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from dataset.augmentation import get_transform
from models.model_factory import build_backbone, build_classifier
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs import cfg, update_config
from models.base_block import FeatClassifier
from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool, time_str
from models.backbone import swin_transformer, resnet, bninception
from PIL import Image

set_seed(605)

def input(image_path, obj_id=None):
    _, valid_tsfm = get_transform(cfg)
    image = Image.open(image_path).convert('RGB')
    image = valid_tsfm(image)
    image = image.unsqueeze(0)
    return image

def KETI(cfg, image):
    nattr = 149
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)

    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)

    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=nattr,
        c_in=c_output,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale =cfg.CLASSIFIER.SCALE
    )

    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    model = get_reload_weight(model_dir, model, pth='KETI_resnet_0.6726672327962699.pkl')
    model.eval()

    with torch.no_grad():
        valid_logits, attns = model(image)

        valid_probs = torch.sigmoid(valid_logits[0])
    return valid_probs

def output(probs):
    probs = probs.detach().cpu().numpy()
    probs = (probs > 0.5).astype(np.int_)

    fl = open('attr_id.json', 'r')
    data = json.load(fl)

    attr_mapping = dict((v, k) for k, v in data.items())

    few = np.where(probs[0] == 1)[0]

    attr_id = []
    for i, item in enumerate(few):
        attr_id.append(attr_mapping[item])
    return attr_id

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str, default="./configs/pedes_baseline/KETI.yaml",

    )
    parser.add_argument("--debug", type=str2bool, default="true")

    args = parser.parse_args()

    return args



args = argument_parser()
update_config(cfg, args)

image = input('22951_0.jpg')
probs = KETI(cfg, image)
result = output(probs)

print(result)


