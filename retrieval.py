# -*- coding: utf-8 -*-
import argparse
import pickle
from backbone.dinov2 import DinoV2
from backbone.vgg import VGGNet
from milvus import MilvusRetrieval
import os
import sys
from os.path import dirname
from PIL import Image
BASE_DIR = dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


class RetrievalEngine(object):

    def __init__(self, index_file, db_name):
        self.index_file = index_file
        self.db_name = db_name
        self.numpy_r = self.faiss_r = self.es_r = self.milvus_r = None

    def milvus_handler(self, query_vector, search_size, threshold):
        # milvus计算
        if self.milvus_r is None:
            self.milvus_r = MilvusRetrieval(self.db_name, self.index_file)
        return self.milvus_r.retrieve(query_vector, search_size, threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size",
                        type=str, default="base",
                        choices=["small", "base", "large", "largest"],
                        help="DinoV2 model type",)
    parser.add_argument("--model_path",
        type=str, default=None,
        help="path to dinov2 model, useful when github is unavailable",
    )
    parser.add_argument("--test_data", type=str, default=os.path.join(BASE_DIR, 'data', 'test', '001_accordion_image_0001.jpg'), help="test data path.")
    parser.add_argument("--index_file", type=str, default=os.path.join(BASE_DIR, 'index', 'train.p'), help="index file path.")
    parser.add_argument("--db_name", type=str, default='image_retrieval', help="database name.")
    parser.add_argument("--engine", type=str, default='milvus', help="retrieval engine.")
    parser.add_argument("--search_size", type=int, default=3, help="retrieval size.")
    parser.add_argument("--threshold", type=float, default=0.3, help="retrieval threshold.")
    args = parser.parse_args()
    # 1.图片推理
    model = DinoV2(args.model_size, args.model_path)
    # model =VGGNet()
    img = Image.open(args.test_data).convert('RGB')
    # query_vector = model.vgg_extract_feat(img)
    query_vector = model.extract_single_image_feature(img)
    # 2.图片检索
    re = MilvusRetrieval(args.db_name)
    result = re.retrieve(query_vector, args.search_size, args.threshold)
    print(result)

