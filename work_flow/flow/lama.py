import torch
from PIL import Image
from work_flow.utils.lama import prepare_img_and_mask
import onnxruntime as ort
import os
import cv2
import numpy as np


class Lama:
    """Segmentation model using SegmentAnything"""

    def __init__(self, model_path, device) -> None:
        self.device = 'cuda' if device == 'GPU' else 'cpu'
        base = os.path.basename(model_path)
        if base.split('.')[1] == "onnx":
            providers = ["CPUExecutionProvider"]
            if device.lower() == "gpu":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 3
            if "OMP_NUM_THREADS" in os.environ:
                sess_opts.inter_op_num_threads = int(
                    os.environ["OMP_NUM_THREADS"]
                )
            sess_options = ort.SessionOptions()
            self.use_onnx = True
            self.ort_session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
        elif base.split('.')[1] == "pt":
            self.model = torch.jit.load(model_path, map_location=self.device)  # 加载 TorchScript 模型
            self.model.eval()  # 设置为评估模式
            self.model.to(self.device)  # 将模型移到指定设备
            self.use_onnx = False
        else:
            raise ValueError(f"Model file {base} is not supported!")

    def pre_process(self, image, inpainting_mask):
        if isinstance(image, Image.Image):
            imagex = image.copy()
        else:
            imagex = Image.fromarray(image)
        if isinstance(inpainting_mask, Image.Image):
            maskx = inpainting_mask.convert("L").copy()  # 确保 mask 是单通道
        else:
            maskx = Image.fromarray(inpainting_mask).convert("L")

        # 调整大小并准备输入
        if self.use_onnx:
            image_resized = imagex.resize((512, 512))
            mask_resized = maskx.resize((512, 512))
            image, inpainting_mask = prepare_img_and_mask(image_resized, mask_resized, 'cpu')
            return {
                'image': image.numpy().astype(np.float32),
                'mask': inpainting_mask.numpy().astype(np.float32)
            }
        else:
            return {'image': image, 'mask': inpainting_mask}

    def post_process(self, masks, image=None):
        """
        Post process masks
        """
        # Find contours

    def predict_shapes(self, image, mask=None):
        """
        Predict shapes from image
        """
        operator = self.pre_process(image, mask)
        if self.use_onnx:
            outputs = self.ort_session.run(None, operator)
            # 处理输出
            output = outputs[0][0]  # 假设输出是第一个元素
            output = output.transpose(1, 2, 0)  # 转换通道顺序为 HWC
            output = output.astype(np.uint8)  # 转换为 uint8 类型
            # 调整输出图像大小，恢复原图尺寸
            inpainting = cv2.resize(output, (image.shape[1], image.shape[0]))

        else:
            # 其他 PyTorch 代码块不变
            if isinstance(image, np.ndarray):
                orig_height, orig_width = image.shape[:2]
            else:
                orig_height, orig_width = np.array(image).shape[:2]
            image, mask = prepare_img_and_mask(image, mask, self.device)
            with torch.inference_mode():
                print(image.shape, mask.shape)
                inpainted = self.model(image, mask)
                cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                inpainting = cur_res[:orig_height, :orig_width]

        return inpainting


