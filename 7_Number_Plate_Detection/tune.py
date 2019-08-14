import pandas as pd
from train import NN
import os
import sys
import zipfile

# self.conv_filters = param_dict['conv_filters']
#         self.kernel_size = (param_dict['kernel_size'], param_dict['kernel_size'])
#         self.pool_size = param_dict['pool_size']
#         self.time_dense_size = param_dict['time_dense_size']
#         self.rnn_size = param_dict['rnn_size']
#         self.batch_size = param_dict['batch_size']

conv_filter_numbers = [8, 16, 32, 64, 128, 256]
kernel_sizes = [3, 5]
time_dense_sizes = [8, 16, 32, 64, 128]
rnn_sizes = [64, 128, 256, 512, 1024, 2048]

result_list = []

result_list.append(['Convolution Filter', 'Kernel Size', 'Time Dense Size', 'Rnn Size', 'Validation Loss'])

best_conv_filter = None
best_kernel = None
best_time_dense_size = None
best_rnn_size = None

min_validation_loss = 1e80

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

vargs = sys.argv[2]
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

for cf in conv_filter_numbers:
    for ks in kernel_sizes:
        for tdn in time_dense_sizes:
            for rnns in rnn_sizes:
                with open('hyperparameters.txt', 'w') as f:
                    f.write('key,valuie\n')
                    # self.conv_filters = param_dict['conv_filters']
                    # self.kernel_size = (param_dict['kernel_size'], param_dict['kernel_size'])
                    # self.pool_size = param_dict['pool_size']
                    # self.time_dense_size = param_dict['time_dense_size']
                    # self.rnn_size = param_dict['rnn_size']
                    # self.batch_size = param_dict['batch_size']
                    f.write('conv_filters,' + str(cf) + '\n')
                    f.write('kernel_size,' + str(ks) + '\n')
                    f.write('pool_size,' + str(2) + '\n')
                    f.write('time_dense_size,' + str(tdn) + '\n')
                    f.write('rnn_size,' + str(rnns) + '\n')
                    f.write('batch_size,' + str(32) + '\n')
                f.close()
                nn = NN()
                nn.setTrainingDataSet(os.path.join(dir_path, 'dataset'))
                nn.setValidationDataSet(os.path.join(dir_path, 'dataset'))
                nn.loadHyperParam('hyperparameters.txt')
                vdl = nn.trainOnlyOCR()
                current_result = [cf, ks, tdn, rnns, vdl[0]]
                result_list.append(current_result)

                if vdl[0] < min_validation_loss:
                    print('val loss reduces from ', min_validation_loss, 'to', vdl[0])
                    min_validation_loss = vdl[0]
                    best_conv_filter = cf
                    best_kernel = ks
                    best_rnn_size = rnns
                    best_time_dense_size = tdn

with open('hyperparameters.txt', 'w') as f:
    f.write('key,valuie\n')
    # self.conv_filters = param_dict['conv_filters']
    # self.kernel_size = (param_dict['kernel_size'], param_dict['kernel_size'])
    # self.pool_size = param_dict['pool_size']
    # self.time_dense_size = param_dict['time_dense_size']
    # self.rnn_size = param_dict['rnn_size']
    # self.batch_size = param_dict['batch_size']
    f.write('conv_filters,' + str(best_conv_filter) + '\n')
    f.write('kernel_size,' + str(best_kernel) + '\n')
    f.write('pool_size,' + str(2) + '\n')
    f.write('time_dense_size,' + str(best_time_dense_size) + '\n')
    f.write('rnn_size,' + str(best_rnn_size) + '\n')
    f.write('batch_size,' + str(32) + '\n')
f.close()

restDF = pd.DataFrame(result_list, None, result_list[0, 1:])
restDF.to_csv('tuning_result.txt')