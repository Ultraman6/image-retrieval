# -*- coding: utf-8 -*-
import os
import pickle

import h5py
from pymilvus import MilvusClient
from pymilvus import IndexType, DataType

THRESHOLD = float(os.environ.get('THRESHOLD', '0.85'))  # 检索阈值

class MilvusRetrieval(object):
    def __init__(self, scheme_name, init=False, dim=None):
        self.client = MilvusClient()
        self.scheme_name = scheme_name

        if init:
            self.create(dim)
        self.client.load_collection(self.scheme_name)

    def create(self, dim):
        if self.scheme_name in self.client.list_collections():
            self.client.drop_collection(collection_name=self.scheme_name)
        # 创建 Schema（启用自动主键）
        schema = MilvusClient.create_schema(
            auto_id=True,  # ✅ 自动生成主键
            enable_dynamic_field=True  # ✅ 允许动态字段
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="service_id", datatype=DataType.INT64)
        schema.add_field(field_name="order_id", datatype=DataType.INT64)
        schema.add_field(field_name="stage", datatype=DataType.INT8)
        schema.add_field(field_name="time", datatype=DataType.VARCHAR)
        schema.add_field(field_name="img_url", datatype=DataType.VARCHAR)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        # schema.add_field(field_name="meta", datatype=DataType.JSON)

        # 配置索引参数（HNSW + IP 相似度）
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="FLAT",  # ✅ 指定索引类型
            metric_type="COSINE",  # ✅ 指定相似度计算方式
            params={"M": 16, "efConstruction": 200}  # ✅ HNSW 参数
        )
        # 创建集合
        self.client.create_collection(
            collection_name=self.scheme_name,
            schema=schema,
            index_params=index_params
        )

    def insert(self, data):
        res_dict = self.client.insert(collection_name=self.scheme_name, data=data)
        print(res_dict['insert_count'], res_dict['cost'])

    def retrieve(self, query_vector, search_size=3, threshold=0.3):
        r_list = []
        results = self.client.search(
            collection_name=self.scheme_name, data=[query_vector],
            limit=search_size,
            output_fields=["service_id", "order_id", "stage", "time", "img_url"],
            search_params={'nprobe': 16}
        )
        for r in results[0]:
            score = float(r['distance'])
            # *0.5 + 0.5
            if score > threshold:
                temp = {
                    "id": r['id'],
                    "service_id": r['entity']['service_id'],
                    "order_id": r['entity']['order_id'],
                    "stage": r['entity']['stage'],
                    "time": r['entity']['time'],
                    "img_url": r['entity']['img_url'],
                    "score": round(score, 6)
                }
                r_list.append(temp)

        return r_list
