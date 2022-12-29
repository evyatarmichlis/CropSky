import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode, Visualizer, _PanopticPrediction
import numpy as np
import cv2
import glob

SKY_ID = 40
MODEL_PATH = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
class Detector:
    def __init__(self,data_set_path,output_path):
        """
        initialize all the required parameters for our task
        :param data_set_path: path for the data set
        :param output_path:  path to the output folder
        """

        #detectron2 CfgNode instance
        self.cfg = get_cfg()
        #I used the panoptic segmentation model
        self.cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_PATH)
        #the treshold
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        #i dont have GPU on my computer so i used cpu - can be replace to cuda if avliable
        self.cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(self.cfg)
        self.data_set_path = data_set_path
        self.output_path = output_path

    def crop_sky(self, image_path):
        """
        remove the sky from the image
        :param image_path: the path to the image we want to crop
        :return: the cropped image
        """
        #read the image and convert to RGB
        image = cv2.imread(image_path, )
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #use the predictor to get the prediction on the image( panoptic segmentation) and the info about each seg
        predictions,segment_info =self.predictor(image)["panoptic_seg"]
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        #get the index of the "sky" class from the metadata of the detection2 model
        index_sky = -1
        for i  ,seg in enumerate(segment_info):
            if seg["category_id"] == SKY_ID:
                index_sky= i
        if index_sky == -1: # didnt found sky in the image
            return image
        #use the index we found to extract the mask of the segment from the model
        pred = _PanopticPrediction(predictions, [segment_info[index_sky]], metadata)
        mask, sinfo = next(pred.semantic_masks())
        #find the location of the first (max in numpy) sky pixel in the y axis
        sky_line = np.max(np.where(mask)[0])
        #cut all of the pixel above this pixel
        crop_image = image[sky_line:, :]
        #return the cropped image
        return crop_image


    def crop_sky_from_data_set(self,):
        "crop the sky from data set"
        files = glob.glob(self.data_set_path)  # path to the data set folder
        for i, f in enumerate(files):
            print(i)
            # use the function we made on image
            image = d.crop_sky(f)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{self.output_path}/croped_images{i}.jpg', image)  # Path to the output folder


if __name__ == '__main__':
    # init detector object
    d = Detector(r'C:\Users\Evyatar\PycharmProjects\sky\sky-dataset\*','croped_images')
    d.crop_sky_from_data_set() # example for use


