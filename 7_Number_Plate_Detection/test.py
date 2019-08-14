import os
import sys
from train import NN
import zipfile

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

data_path = dir_path

data_zip_file = zipfile.ZipFile(file_path, 'r')
print(file_path)
print('Extracting Test Data...')
data_zip_file.extractall(dir_path)

data_zip_file.close()

model1_path = sys.argv[2]
model2_path = sys.argv[3]

nn = NN()
nn.loadHyperParam('hyperparameters.txt')
nn.load_saved_weight_path(model1_path, model2_path)
yolo_mAP, ocr_loss = nn.test(os.path.join(data_path, 'dataset'))
print()
print()
print('=====================================================================================')
print('Evaluation complete on Test Data Set')
print('First Stage Yolo: mean Average Precision(mAP):', yolo_mAP)
print('Second Stage OCR: Connectionist temporal classification(ctc) Loss:', ocr_loss)
print('=======================================================================================')

