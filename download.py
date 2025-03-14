import os
import mysql.connector
from urllib.parse import urlparse
from mysql.connector import Error
import json
import logging
import cv2
import numpy as np
import requests
from database_using import *
from work_flow.flows.ppocr_v4_lama import PPOCRv4LAMA
from backbone.dinov2 import DinoV2
from milvus import MilvusRetrieval

jdbc_url = "mysql://47.99.65.68:3306/yanglao"
host = '47.99.65.68'
database = 'yanglao'
username = "dhgxjbgs"
password = "D23@#hGb"

time_map = {
    'start_img': 'start_time',
    'img_url': 'create_time',
    'end_img': 'end_time'
}

model = DinoV2('base', None)
with open('E:\GitHub\yanglao_sys\yoloWorld_detectSeg_backend\scripts\model_config.json', 'r', encoding='utf-8') as f:
    model_config = json.load(f)
ocr = PPOCRv4LAMA(model_config, logging.info)
re = MilvusRetrieval('history_photos', init=True, dim=model.dim)

def get_img_urls(specific_service_id, img_stages, start_data, end_data):
    # work_order_query = f"-- SELECT order_id FROM work_order WHERE service_id = {specific_service_id} and DATE(start_time) >= {start_data} and DATE(start_time) <= {end_data}"
    work_order_query = "SELECT order_id, {0} time FROM work_order WHERE  service_id = {1} AND start_time >= \"{2}\" and start_time <= \"{3}\""
    service_log_query_template = "SELECT {0} FROM service_log WHERE order_id = {1}"

    conn = None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=username,
            password=password,
        )

        if conn.is_connected():
            cursor = conn.cursor()

            urls_all = dict()
            # 三层列表，用于存储所有订单的图片 URL
            for attribute in img_stages:
                cursor.execute(work_order_query.format(time_map[attribute], specific_service_id,
                                                       start_data, end_data))
                order_ids_times = [(row[0], row[1]) for row in cursor.fetchall()]
            for item in order_ids_times:
                order_id = item[0]
                stage_time = item[1]
                for attribute in img_stages:
                    key = (order_id, attribute, stage_time)
                    service_log_query = service_log_query_template.format(attribute, order_id)
                    cursor.execute(service_log_query)
                    urls_all[key] = []
                    for (urls_string,) in cursor.fetchall():
                        if urls_string:
                            urls = urls_string.split(',')
                            urls_all[key] += urls
                            # download_images(order_id, attribute, urls, base_dir)
                        else:
                            print(f"No URLs found for order_id {order_id} and attribute {attribute}")
            return urls_all

    except Error as e:
        print(f"Error: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

def get_file_name_from_url(url):
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path)

def img_url_to_np(img_url):
    response = requests.get(img_url)
    if response.status_code == 200:
        # 使用 OpenCV 读取图片并转换为 NumPy 数组
        img = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # 解码为图片格式
        size = img.nbytes / 1024  # 当前图片大小（KB）
        return img, size
    else:
        raise Exception(f"Failed to fetch image from URL: {img_url}")

def handle_before_add(img_url):
    img_np, size = img_url_to_np(img_url)
    processed_img = model.predict_shapes(img_np).image
    feat = model.extract_single_image_feature(processed_img)
    return feat

def add_img_by_url(vec, img_url, service_id, order_id, stage, stage_time):
    item = {
        'vector': vec,
        "service_id": service_id,
        "order_id": order_id,
        "img_url": img_url,
        "stage": stage,
        "time": stage_time
    }
    re.insert([item,])

if __name__ == '__main__':
    start_data = '2024-01-01 00:00:00'
    end_data = '2024-04-30 23:59:59'

    service_id = 406
    # start_order_id = 445209
    urls_all = get_img_urls(service_id, ["start_img", "img_url", "end_img"], start_data, end_data)
    for item, urls in urls_all.items():
        for url in urls:
            vec = handle_before_add(url)
            add_img_by_url(vec, url, str(service_id), str(item[0]), str(item[1]), str(item[2]))

