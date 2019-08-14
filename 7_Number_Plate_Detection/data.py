import os
import random
import shutil
import sys
import zipfile

root_dir = None

cwd = os.getcwd()


def generateZipFile(filepath, zipFileNameList, annotation_list, ocr_label_file, ocr_labels, st, en):
    print('Generating', filepath, '...')
    print(st, en)
    splitZipFile = zipfile.ZipFile(filepath, 'w')
    print('\tCopying Images...')
    for filename in zipFileNameList:
        file = zpfile.open(filename)

        if not (filename in annotation_list or filename == ocr_label_file):
            splitZipFile.write(filename)
    annotation_list_len = len(annotation_list)
    ocr_labels_len = len(ocr_labels)

    print('\tGenerating Annotation')

    for file in annotation_list[int(st * annotation_list_len):int(en * annotation_list_len)]:
        splitZipFile.write(file)

    print('\tGenerating Labels')

    with open('dataset/number_label.txt', 'w')  as f:
        for label in ocr_labels[int(st * ocr_labels_len):int(en * ocr_labels_len)]:
            s = str(label) + '\n'
            s = s.replace('b\'', '')
            s = s.replace('\'', '')
            f.write(s)
    f.close()
    splitZipFile.write('dataset/number_label.txt')
    splitZipFile.close()


args = sys.argv[1]
args = args.replace('.\\', '')
file_name = args.split('\\')[-1]
dir_name = args[:args.find(file_name) - 1]

print(file_name)
print(dir_name)

root_dir = os.path.join(cwd, dir_name)
root_file_path = os.path.join(root_dir, file_name)

print(root_file_path)

print('Creating Directories...')

dirList = ['Test', 'Train', 'Validation']
subDirDict = {'Test': ['Test_10_percent'],
              'Train': ['Best_hyperparameter_80_percent', 'Under_10_min_training', 'Under_90_min_tuning'],
              'Validation': ['Validation_10_percent']}
percentange = {'Test_10_percent': [0, .1], 'Best_hyperparameter_80_percent': [.1, .9], 'Under_90_min_tuning': [.4, .8],
               'Under_10_min_training': [.15, .35],
               'Validation_10_percent': [.9, 1]}

for d in dirList:
    temp = os.path.join(root_dir, d)
    if not os.path.isdir(temp):
        os.mkdir(temp)
    for sub_dir in subDirDict[d]:
        temp = os.path.join(root_dir, d, sub_dir)
        if not os.path.isdir(temp):
            os.mkdir(temp)

zpfile = zipfile.ZipFile(root_file_path, 'r')

print('Extracting Data...')
zpfile.extractall(cwd)

print('Reading Data...')

zpFileNameList = zpfile.namelist()

annotation_list = []

ocr_label_file = None

for filename in zpFileNameList:
    if 'annotation_97c' in filename and '.' in filename:
        annotation_list.append(filename)
    if 'number_label.txt' in filename:
        ocr_label_file = filename

label_file = zpfile.open(ocr_label_file)

ocr_labels = label_file.read().splitlines()

random.shuffle(annotation_list)
random.shuffle(ocr_labels)

for d in dirList:
    for sub_dir in subDirDict[d]:
        zip_file_name = os.path.join(root_dir, d, sub_dir, 'data.zip')

        generateZipFile(zip_file_name, zpFileNameList, annotation_list, ocr_label_file,
                        ocr_labels, percentange[sub_dir][0],
                        percentange[sub_dir][1])

print('removing temporary files...')
shutil.rmtree(os.path.join(cwd, 'dataset'))
