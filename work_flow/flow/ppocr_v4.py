import os
import onnxruntime as ort
from work_flow.utils.ppocr.text_system import TextSystem

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class PPOCRv4:

    def __init__(self, det_model_path, rec_model_path, cls_model_path, device, drop_score=0.5,
                 det_algorithm='DB', rec_algorithm='SVTR_LCNet', use_angle_cls=False, lang='ch') -> None:

        self.det_net = self.load_model(det_model_path, device)
        self.rec_net = self.load_model(rec_model_path, device)
        self.cls_net = self.load_model(cls_model_path, device)
        self.det_algorithm = det_algorithm
        self.rec_algorithm = rec_algorithm
        self.drop_score = drop_score
        self.use_angle_cls = use_angle_cls
        self.current_dir = os.path.dirname(__file__)
        self.lang = lang
        if self.lang == "ch":
            self.rec_char_dict = "ppocr_keys_v1.txt"
        elif self.lang == "japan":
            self.rec_char_dict = "japan_dict.txt"

        self.args = self.parse_args()
        self.text_sys = TextSystem(self.args)

    def parse_args(self):
        args = Args(  # 关键参数直接写死，你妈是不是死了？
            use_onnx=True,
            # params for prediction engine
            use_gpu=True,
            use_xpu=False,
            use_npu=False,
            ir_optim=True,
            use_tensorrt=False,
            min_subgraph_size=15,
            precision="fp32",
            gpu_mem=500,
            gpu_id=0,
            # params for text detector
            page_num=0,
            det_algorithm=self.det_algorithm,
            det_model=self.det_net,
            det_limit_side_len=960,
            det_limit_type="max",
            det_box_type="quad",
            # DB parmas
            det_db_thresh=0.3,
            det_db_box_thresh=0.6,
            det_db_unclip_ratio=1.5,
            max_batch_size=10,
            use_dilation=False,
            det_db_score_mode="fast",
            # EAST parmas
            det_east_score_thresh=0.8,
            det_east_cover_thresh=0.1,
            det_east_nms_thresh=0.2,
            # SAST parmas
            det_sast_score_thresh=0.5,
            det_sast_nms_thresh=0.2,
            # PSE parmas
            det_pse_thresh=0,
            det_pse_box_thresh=0.85,
            det_pse_min_area=16,
            det_pse_scale=1,
            # FCE parmas
            scales=[8, 16, 32],
            alpha=1.0,
            beta=1.0,
            fourier_degree=5,
            # params for text recognizer
            rec_algorithm=self.rec_algorithm,
            rec_model=self.rec_net,
            rec_image_inverse=True,
            rec_image_shape="3, 48, 320",
            rec_batch_num=6,
            max_text_length=25,
            rec_char_dict_path=os.path.join(
                self.current_dir, f"../configs/ppocr/{self.rec_char_dict}"
            ),
            use_space_char=True,
            drop_score=self.drop_score,
            # params for e2e
            e2e_algorithm="PGNet",
            e2e_model_dir="",
            e2e_limit_side_len=768,
            e2e_limit_type="max",
            # PGNet parmas
            e2e_pgnet_score_thresh=0.5,
            e2e_char_dict_path=os.path.join(
                self.current_dir, "../configs/ppocr/ppocr_ic15_dict.txt"
            ),
            e2e_pgnet_valid_set="totaltext",
            e2e_pgnet_mode="fast",
            # params for text classifier
            use_angle_cls=self.use_angle_cls,
            cls_model=self.cls_net,
            cls_image_shape="3, 48, 192",
            label_list=["0", "180"],
            cls_batch_num=6,
            cls_thresh=0.9,
            enable_mkldnn=False,
            cpu_threads=10,
            use_pdserving=False,
            warmup=False,
            # SR parmas
            sr_model_dir="",
            sr_image_shape="3, 32, 128",
            sr_batch_num=1,
        )
        return args

    def load_model(self, model_path, device):

        model_task = os.path.splitext(
            model_path
        )[0]
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Could not download or initialize {model_task} model."
            )

        self.sess_opts = ort.SessionOptions()
        if "OMP_NUM_THREADS" in os.environ:
            self.sess_opts.inter_op_num_threads = int(
                os.environ["OMP_NUM_THREADS"]
            )
        self.providers = ["CPUExecutionProvider"]

        if device == "gpu":
            self.providers = ["CUDAExecutionProvider"]
        net = ort.InferenceSession(
            model_path,
            providers=self.providers,
            sess_options=self.sess_opts,
        )
        return net
