# -*- coding: utf-8 -*-
import pickle
import argparse
from backbone.dinov2 import DinoV2
from backbone.vgg import VGGNet
import os
import sys
from os.path import dirname
from PIL import Image
from milvus import MilvusRetrieval
BASE_DIR = dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size",
                        type=str, default="base",
                        choices=["small", "base", "large", "largest"],
                        help="DinoV2 model type",)
    parser.add_argument("--model_path",
        type=str, default=None,
        help="path to dinov2 model, useful when github is unavailable",
    )
    parser.add_argument("--train_data", type=str, default=os.path.join(BASE_DIR, 'data', 'train'), help="train data path.")
    parser.add_argument("--db_name", type=str, default='image_retrieval', help="database name.")
    args = parser.parse_args()
    img_list = get_imlist(args.train_data)
    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    items = []
    model = DinoV2(args.model_size, args.model_path)
    # model = VGGNet()
    for i, img_path in enumerate(img_list):
        img = Image.open(img_path).convert('RGB')
        norm_feat = model.extract_single_image_feature(img)
        print(len(norm_feat))
        img_name = str(os.path.basename(img_path))
        items.append({'vector': norm_feat, 'meta': img_name})
        print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))
    print("--------------------------------------------------")
    print("         writing feature extraction results")
    print("--------------------------------------------------")
    # pickle.dump(key_value, open(args.index_file, 'wb'))
    re = MilvusRetrieval(args.db_name, init=True, dim=model.dim)
    re.insert(items)
