from torch.utils.data.dataset import ConcatDataset
from .JointsDataset import JointsDataset


class CustomDataset(JointsDataset):
    
    def __init__(self, cfg, root, image_set, transform) -> None:
        super().__init__(cfg, root, image_set, False, transform)
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.DEEPFASHION2_BBOX_FILE
        # self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.mini_dataset = cfg.DATASET.MINI_DATASET
        self.select_cat = cfg.DATASET.SELECT_CAT
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        
        
    def __getitem__(self, index):
        return super().__getitem__(index)

    
    def __len__():
        pass