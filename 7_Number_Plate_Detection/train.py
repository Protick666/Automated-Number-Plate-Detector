import os
import numpy as np
from Lib.preprocessing import parse_annotation
from Lib.frontend import YOLO
import sys
import zipfile
import pandas as pd
from Lib.ocr import OCR_NN


class NN:

    def __init__(self):
        self.backend = 'Full Yolo'
        self.input_size = 768
        self.labels = ['WhiteNumberPlate']
        self.max_box_per_image = 10
        self.anchors = [3.52, 2.83, 3.58, 2.62, 3.61, 2.43, 3.83, 2.75, 3.91, 2.52]
        self.yolo = YOLO(backend=self.backend,
                         input_size=self.input_size,
                         labels=self.labels,
                         max_box_per_image=self.max_box_per_image,
                         anchors=self.anchors)
        self.train_imgs = None
        self.valid_imgs = None
        self.train_times = 8
        self.valid_times = 1
        self.nb_epochs = 1
        self.learning_rate = 1e-4
        self.batch_size = 1
        self.warmup_epochs = 4
        self.object_scale = 5.0
        self.no_object_scale = 1.0
        self.coord_scale = 1.0
        self.class_scale = 1.0
        self.saved_weights_name = 'model1.h5'
        self.debug = False
        self.train_labels = None
        self.valid_labels = None

        self.ocr_train_root_dir = 'Lib/temp'
        self.ocr_valid_root_dir = 'Lib/temp'
        ocrdf = pd.read_csv('Lib/temp/number_label.txt')
        self.ocr_train_annotation = ocrdf.as_matrix()
        self.ocr_valid_annotation = ocrdf.as_matrix()
        self.ocr = OCR_NN(128, self.ocr_train_annotation, self.ocr_valid_annotation,
                          self.ocr_train_root_dir, self.ocr_train_root_dir, False)

    def setTrainingDataSet(self, data_path):
        img_dir = os.path.join(data_path, 'train_97c/')
        annotation_dir = os.path.join(data_path, 'annotation_97c')
        img_dir += "/"
        annotation_dir += "/"
        self.train_imgs, self.train_labels = parse_annotation(annotation_dir, img_dir, self.labels)
        ocrdf = pd.read_csv(os.path.join(data_path, 'number_label.txt'))
        self.ocr_train_root_dir = data_path
        self.ocr_train_annotation = ocrdf.as_matrix()
        self.ocr.setTrainingData(self.ocr_train_annotation, self.ocr_train_root_dir)

    def setValidationDataSet(self, data_path):
        img_dir = os.path.join(data_path, 'train_97c/')
        annotation_dir = os.path.join(data_path, 'annotation_97c')
        img_dir += "/"
        annotation_dir += "/"
        self.valid_imgs, self.valid_labels = parse_annotation(annotation_dir, img_dir, self.labels)
        ocrdf = pd.read_csv(os.path.join(data_path, 'number_label.txt'))
        self.ocr_valid_root_dir = data_path
        self.ocr_valid_annotation = ocrdf.as_matrix()
        self.ocr.setValidationData(self.ocr_valid_annotation, self.ocr_valid_root_dir)

    def load_saved_weight(self):
        self.yolo.load_weights(self.saved_weights_name)
        self.ocr.build_model()
        self.ocr.load_saved_model()

    def load_saved_weight_path(self, path1, path2):
        self.saved_weights_name = path1
        self.ocr.model_name = path2
        self.load_saved_weight()

    def loadHyperParam(self, param_file):
        df = pd.read_csv(param_file)
        mt = df.as_matrix()

        param = {}
        for data in mt:
            param[data[0]] = data[1]
        #print(param)
        self.ocr.setNetworkParam(param)

    def train(self):
        print('Training OCR...')
        self.ocr.build_model()
        self.ocr.train_model()
        print('Training Yolo...')
        self.yolo.train(train_imgs=self.train_imgs,
                        valid_imgs=self.valid_imgs,
                        train_times=self.train_times,
                        valid_times=self.valid_times,
                        nb_epochs=self.nb_epochs,
                        learning_rate=self.learning_rate,
                        batch_size=self.batch_size,
                        warmup_epochs=self.warmup_epochs,
                        object_scale=self.object_scale,
                        no_object_scale=self.no_object_scale,
                        coord_scale=self.coord_scale,
                        class_scale=self.class_scale,
                        saved_weights_name=self.saved_weights_name,
                        debug=self.debug)

    def trainOnlyOCR(self):
        self.ocr.build_model()
        validation_loss = self.ocr.train_model()
        return validation_loss

    def test(self, data_path):
        img_dir = os.path.join(data_path, 'train_97c/')
        annotation_dir = os.path.join(data_path, 'annotation_97c')
        img_dir += "/"
        annotation_dir += "/"
        test_imgs, test_labels = parse_annotation(annotation_dir, img_dir, self.labels)
        mAP = self.yolo.test(test_imgs)
        print('Yolo Test Result, mAP:', mAP)

        ocrdf = pd.read_csv(os.path.join(data_path, 'number_label.txt'))
        test_annotation = ocrdf.as_matrix()
        test_ds = self.ocr.getTestDataSet(test_annotation, data_path)
        test_loss=self.ocr.test_model(test_ds)
        print('OCR Test Loss:',test_loss)

        return mAP[0], test_loss


def trainFullDataSet():
    cwd = os.getcwd()
    args = sys.argv[1]
    print(args)
    args = args.replace('.\\', '')
    print(args)
    file_name = args.split('\\')[-1]
    dir_name = args[:args.find(file_name) - 1]

    print(file_name)
    print(dir_name)
    dir_path = os.path.join(cwd, dir_name)
    file_path = os.path.join(dir_path, file_name)

    data_zip_file = zipfile.ZipFile(file_path, 'r')
    print(file_path)
    print('Extracting Training Data...')
    data_zip_file.extractall(dir_path)

    data_zip_file.close()

    vargs = '.\Data\Validation\Validation_10_percent\data.zip'
    print(vargs)
    vargs = vargs.replace('.\\', '')
    print(vargs)
    vfile_name = vargs.split('\\')[-1]
    vdir_name = vargs[:vargs.find(vfile_name) - 1]

    print(vfile_name)
    print(vdir_name)
    vdir_path = os.path.join(cwd, vdir_name)
    vfile_path = os.path.join(vdir_path, vfile_name)

    print(vfile_path)

    vdata_zip_file = zipfile.ZipFile(vfile_path, 'r')
    print('Extracting Validation Data...')
    vdata_zip_file.extractall(vdir_path)
    vdata_zip_file.close()

    print('ok')

    paramfile = sys.argv[2]
    paramfile = paramfile.replace('.\\', '')
    paramfile_path = os.path.join(cwd, paramfile)

    nn = NN()
    nn.setTrainingDataSet(os.path.join(dir_path, 'dataset'))
    nn.setValidationDataSet(os.path.join(dir_path, 'dataset'))
    nn.loadHyperParam(paramfile_path)
    nn.train()


if __name__ == '__main__':
    trainFullDataSet()
