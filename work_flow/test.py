import argparse
import os
import cv2
from work_flow import *
BASE_DIR = "/mnt/d/github/image_retrieval"
parser = argparse.ArgumentParser()
parser.add_argument("--face_model_path", type=str, default=os.path.join(BASE_DIR, 'backbone', 'models', 'yolov6lite_l_face.onnx'))
parser.add_argument("--det_model_path", type=str, default=os.path.join(BASE_DIR, 'backbone', 'models', 'ch_PP-OCRv4_det_infer.onnx'))
parser.add_argument("--rec_model_path", type=str, default=os.path.join(BASE_DIR, 'backbone', 'models', 'ch_PP-OCRv4_rec_infer.onnx'))
parser.add_argument("--cls_model_path", type=str, default=os.path.join(BASE_DIR, 'backbone', 'models', 'ch_ppocr_mobile_v2.0_cls_infer.onnx'))
parser.add_argument("--lama_model_path", type=str, default=os.path.join(BASE_DIR, 'backbone', 'models', 'lama_fp32.onnx'))
parser.add_argument("--device", type=str, default="gpu")
args = parser.parse_args()

face_extractor = YOLOv6Face(args.face_model_path, args.device)
watermark_remover = PPOCRv4LAMA(args)
face1 = cv2.imread("data/order/face1.jpg")
water1 = cv2.imread("data/order/water1.jpg")

faces = face_extractor.predict_shapes(face1)
water = watermark_remover.predict_shapes(water1)
for face in faces:
    cv2.imshow("face", face)
cv2.imshow("water", water1)



