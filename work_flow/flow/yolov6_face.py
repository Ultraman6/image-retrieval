import cv2
import numpy as np
from work_flow.base.build_onnx_engine import OnnxBaseModel
from work_flow.utils.yolov6_face import letterbox, xywh2xyxy, warp_im, coord5point1, coord5point2, imgSize1, imgSize2, numpy_nms

class YOLOv6Face:

    def __init__(self, model_path, device='gpu', scale='112x112', conf_thres=0.45, iou_thres=0.25, agnostic=False, input_shape=(320,320)) -> None:
        self.scale = scale
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic = agnostic
        self.input_shape = input_shape
        self.net = OnnxBaseModel(model_path, device)

    def preprocess(self, image, upsample_mode="letterbox"):
        self.img_height, self.img_width = image.shape[:2]
        # Upsample
        if upsample_mode == "resize":
            input_img = cv2.resize(
                image, self.input_shape
            )
        elif upsample_mode == "letterbox":
            input_img = letterbox(image, self.input_shape)[0]
        elif upsample_mode == "centercrop":
            m = min(self.img_height, self.img_width)
            top = (self.img_height - m) // 2
            left = (self.img_width - m) // 2
            cropped_img = image[top : top + m, left : left + m]
            input_img = cv2.resize(
                cropped_img, self.input_shape
            )
        else:
            raise NotImplementedError('Upsample mode not implemented')
        # Transpose
        input_img = input_img.transpose(2, 0, 1)
        # Expand
        input_img = input_img[np.newaxis, :, :, :].astype(np.float32)
        # Contiguous
        input_img = np.ascontiguousarray(input_img)
        # Norm
        blob = input_img / 255.0
        return blob

    def postprocess(
        self,
        prediction,
        multi_label=False,
        max_det=1000,
    ):
        """
        Post-process the network's output, to get the
        bounding boxes, key-points and their confidence scores.
        """

        """Runs Non-Maximum Suppression (NMS) on inference results.
        Args:
            prediction: (tensor), with shape [N, 15 + num_classes], N is the number of bboxes.
            multi_label: (bool), when it is set to True, one box can have multi labels,
                otherwise, one box only huave one label.
            max_det:(int), max number of output bboxes.
        Returns:
            list of detections, echo item is one tensor with shape (num_boxes, 16),
                16 is for [xyxy, ldmks, conf, cls].
        """
        num_classes = prediction.shape[2] - 15  # number of classes
        pred_candidates = np.logical_and(
            prediction[..., 14] > self.conf_thres,
            np.max(prediction[..., 15:], axis=-1) > self.conf_thres,
        )

        # Function settings.
        max_wh = 4096  # maximum box width and height
        max_nms = (
            30000  # maximum number of boxes put into torchvision.ops.nms()
        )
        multi_label &= num_classes > 1  # multiple labels per box

        output = [np.zeros((0, 16))] * prediction.shape[0]

        for img_idx, x in enumerate(
            prediction
        ):  # image index, image inference
            x = x[pred_candidates[img_idx]]  # confidence

            # If no box remains, skip the next process.
            if not x.shape[0]:
                continue

            # confidence multiply the objectness
            x[:, 15:] *= x[:, 14:15]  # conf = obj_conf * cls_conf

            # (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix's shape is  (n,16), each row represents (xyxy, conf, cls, lmdks)
            if multi_label:
                box_idx, class_idx = np.nonzero(x[:, 15:] > self.conf_thres).T
                x = np.concatenate(
                    (
                        box[box_idx],
                        x[box_idx, class_idx + 15, None],
                        class_idx[:, None].astype(np.float32),
                        x[box_idx, 4:14],
                    ),
                    1,
                )
            else:
                conf = np.max(x[:, 15:], axis=1, keepdims=True)
                class_idx = np.argmax(x[:, 15:], axis=1, keepdims=True)
                x = np.concatenate(
                    (box, conf, class_idx.astype(np.float32), x[:, 4:14]), 1
                )[conf.ravel() > self.conf_thres]

            # Check shape
            num_box = x.shape[0]  # number of boxes
            if not num_box:  # no boxes kept.
                continue
            elif num_box > max_nms:  # excess max boxes' number.
                x = x[
                    x[:, 4].argsort(descending=True)[:max_nms]
                ]  # sort by confidence

            # Batched NMS
            class_offset = x[:, 5:6] * (
                0 if self.agnostic else max_wh
            )  # classes
            boxes, scores = (
                x[:, :4] + class_offset,
                x[:, 4],
            )  # boxes (offset by class), scores

            keep_box_idx = numpy_nms(boxes, scores, self.iou_thres)  # NMS
            if keep_box_idx.shape[0] > max_det:  # limit detections
                keep_box_idx = keep_box_idx[:max_det]

            output[img_idx] = x[keep_box_idx]

        return output

    def crop_transform(self, img_im, bounding_box, points):
        shape = img_im.shape
        height = shape[0]
        width = shape[1]
        x1, y1, x2, y2 = bounding_box
        # 外扩大100%，防止对齐后人脸出现黑边
        new_x1 = max(int(1.50 * x1 - 0.50 * x2), 0)
        new_x2 = min(int(1.50 * x2 - 0.50 * x1), width - 1)
        new_y1 = max(int(1.50 * y1 - 0.50 * y2), 0)
        new_y2 = min(int(1.50 * y2 - 0.50 * y1), height - 1)

        # 得到原始图中关键点坐标
        left_eye_x, left_eye_y = points[:2]
        right_eye_x, right_eye_y = points[2:4]
        nose_x, nose_y = points[4:6]
        left_mouth_x, left_mouth_y = points[6:8]
        right_mouth_x, right_mouth_y = points[8:10]

        # 得到外扩100%后图中关键点坐标
        new_left_eye_x = left_eye_x - new_x1
        new_right_eye_x = right_eye_x - new_x1
        new_nose_x = nose_x - new_x1
        new_left_mouth_x = left_mouth_x - new_x1
        new_right_mouth_x = right_mouth_x - new_x1
        new_left_eye_y = left_eye_y - new_y1
        new_right_eye_y = right_eye_y - new_y1
        new_nose_y = nose_y - new_y1
        new_left_mouth_y = left_mouth_y - new_y1
        new_right_mouth_y = right_mouth_y - new_y1

        face_landmarks = [[new_left_eye_x, new_left_eye_y],  # 在扩大100%人脸图中关键点坐标
                          [new_right_eye_x, new_right_eye_y],
                          [new_nose_x, new_nose_y],
                          [new_left_mouth_x, new_left_mouth_y],
                          [new_right_mouth_x, new_right_mouth_y]]
        face = img_im[new_y1: new_y2, new_x1: new_x2]  # 扩大100%的人脸区域

        if self.scale == '112x96':
            dst = warp_im(face, face_landmarks, coord5point1)  # 112x96对齐后尺寸
            crop_im = dst[0:imgSize1[0], 0:imgSize1[1]]
        elif self.scale == '112x112':
            dst = warp_im(face, face_landmarks, coord5point2)  # 112x112对齐后尺寸
            crop_im = dst[0:imgSize2[0], 0:imgSize2[1]]
        else:
            raise ValueError("scale must be 112x96 or 112x112")

        return crop_im

    def predict_shapes(self, image):
        """
        Predict shapes from image
        """

        if image is None:
            return []
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blob = self.preprocess(input_image)
        predictions = self.net.get_ort_inference(blob)
        results = self.postprocess(predictions)[0]

        faces = []
        for i, r in enumerate(reversed(results)):
            xyxy, score, cls_id, lmdks = r[:4], r[4], r[5], r[6:]
            if score < self.conf_thres:
                continue
            lmdks = list(map(int, lmdks))

            if self.scale is not None:
                crop_img = self.crop_transform(image.copy(), xyxy, lmdks)
                faces.append({'img': crop_img, 'score': score})

        return faces

