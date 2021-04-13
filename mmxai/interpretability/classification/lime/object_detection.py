#from detectron2.utils.logger import setup_logger
#setup_logger()
import numpy as np
import os, json, cv2, random
from mmxai.interpretability.classification.lime.detectron2 import model_zoo
from mmxai.interpretability.classification.lime.detectron2.engine import DefaultPredictor
from mmxai.interpretability.classification.lime.detectron2.config import get_cfg
from mmxai.interpretability.classification.lime.detectron2.utils.visualizer import Visualizer, _create_text_labels
from mmxai.interpretability.classification.lime.detectron2.data import MetadataCatalog, DatasetCatalog


def object_detection_predictor():
    cfg = get_cfg()
    #config_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    #config_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    config_path = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "./checkpoint.pkl"  ########### the path to the checkpoint
    predictor = DefaultPredictor(cfg) #
    return predictor,cfg


def object_detection_obtain_label(predictor,cfg,img):

    outputs = predictor(img)
    predictions = outputs["instances"]
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    labels = _create_text_labels(classes, None, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes", None))
    label = np.unique(np.array(labels))
    return label


def compare_labels(o_label, n_label):
    o_label_num = o_label.shape[0]
    n_label_num = n_label.shape[0]
    n_label_np = np.zeros(o_label_num)
    for i in range(n_label_num):
        for j in range(o_label_num):
            if n_label[i] == o_label[j]:
                n_label_np[j] = 1

    return n_label_np
