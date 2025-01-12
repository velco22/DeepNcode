import argparse
import random
import time

import numpy as np

import datasets
import models
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, L1Loss
from torchvision import transforms
import torchvision
from xlwt import Workbook

# from hwcounter import Timer, count, count_end

from scipy.spatial.distance import hamming

## This script is adapted from the following public repository:
## https://github.com/adnansirajrakin/T-BFA

parser = argparse.ArgumentParser(description='Overhead on DNNs')
parser.add_argument('--type', type=int, default=4, help='type of the attack')
parser.add_argument('--data_dir', type=str, default='data/')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset for processing')
parser.add_argument('--num_classes', '-c', default=10, type=int, help='number of classes in the dataset')
parser.add_argument('--arch', '-a', type=str, default='resnet20_quan', help='model architecture')
parser.add_argument('--bits', type=int, default=8, help='quantization bits')
parser.add_argument('--n', type=int, default=1, help='avrg of how many rounds')
parser.add_argument('--ocm', action='store_true', help='output layer coding with bit strings')
parser.add_argument('--output_act', type=str, default='linear', help='output act. (only linear and tanh is supported)')
parser.add_argument('--code_length', '-cl', default=16, type=int, help='length of codewords')
parser.add_argument('--outdir', type=str, default='results/', help='folder where the model is saved')
parser.add_argument('--batch', '-b', default=128, type=int, metavar='N', help='Mini-batch size (default: 128)')
parser.add_argument('--gpu', default="0", type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--iters', type=int, default=5000, help='max attack iterations (def: 5000)')
parser.add_argument('--source_start', type=int, default=0, help='source_start')
parser.add_argument('--source_end', type=int, default=50, help='source_end')
parser.add_argument('--avgs', type=int, default=5, help='average of how many rounds')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if not os.path.exists("tbfa_results/"):
    os.makedirs("tbfa_results/")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
gpu_list = [int(i) for i in args.gpu.strip().split(",")] if args.gpu is not "0" else [0]
if args.gpu == "-1":
    device = torch.device('cpu')
    print('Using cpu')
else:
    device = torch.device('cuda')
    print('Using gpu: ' + args.gpu)

def validate(model, loader, C):
    model.eval()
    correct = 0
    time_array = list()
    with torch.no_grad():

        time_eva_all_samples_start = time.time()

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            time_eva_one_sample_start = time.time()

            output = nn.Softmax()(model(data))

            time_eva_one_sample_end = time.time()

            time_array.append(time_eva_one_sample_end - time_eva_one_sample_start)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        time_eva_all_samples_end = time.time()

    print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, loader.sampler.__len__(), 100. * correct / loader.sampler.__len__()))

    # acc, time_all, time_one_avr, arr, arr_avrg
    return 100. * correct / loader.sampler.__len__(), time_eva_all_samples_end - time_eva_all_samples_start, (time_eva_all_samples_end - time_eva_all_samples_start) / loader.sampler.__len__(), time_array, np.array(time_array).sum() / loader.sampler.__len__()

def get_all_weights(model):
    weights_get = np.array([], dtype=np.ndarray)
    for m in model.cpu().modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            weights_get = np.append(weights_get, m.weight.data.cpu().numpy().flatten())
    return weights_get


# function for 2 arrays comparison and return the different elements
def compare_weights(weights):
    diff = np.empty((weights.shape[0], 2), dtype=np.ndarray)
    for i in range(weights.shape[0]):
        for z, y in zip(weights[i][0], weights[i][1]):
            if z != y:
                if diff[i][0] is None:
                    diff[i][0] = np.array([z])
                    diff[i][1] = np.array([y])
                else:
                    diff[i][0] = np.concatenate([diff[i][0], np.array([z])], axis=0)
                    diff[i][1] = np.concatenate([diff[i][1], np.array([y])], axis=0)

    return diff


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Load dataset
    DATASET = datasets.__dict__[args.dataset](args)
    train_loader, test_loader = DATASET.loaders()

    assert args.output_act == 'linear'
    n_output = args.num_classes
    criterion = CrossEntropyLoss()
    C = torch.tensor(np.eye(args.num_classes)).to(device)

    # Evaluate clean accuracy
    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        model = models.__dict__[args.arch](n_output, args.bits, args.output_act)
    elif args.dataset == 'ImageNet':
        model, state_dict = models.__dict__[args.arch](n_output, args.bits)

    # model = models.__dict__[args.arch](n_output, args.bits, args.output_act)

    model = nn.DataParallel(model, gpu_list).to(device) if len(gpu_list) > 1 else nn.DataParallel(model).to(device)

    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        # model.load_state_dict(torch.load(args.outdir + 'model_best.pth.tar', map_location=device)['state_dict'])
        state_dict = torch.load(args.outdir + 'model_best.pth.tar', map_location=device)['state_dict']
        model.load_state_dict(state_dict)

    # print(model)

    weight_conversion(model)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters: {total_params}")

    par = count_parameters(model)
    print(f"Number of parameters: {par}")

    beginning_test_acc, time_all, time_one_avr, time_arr, time_arr_avr = validate(model, test_loader, C)
    print("beginning_test_acc: ", beginning_test_acc)
    # print("time_all:         ", time_all)
    print("time_arr_all:     ", np.array(time_arr).sum())
    # print("time_one_avr:     ", time_one_avr)
    print("time_arr_one_avr: ", time_arr_avr)
    # print("time_one:         ", time_one)
    print("--"*20)
    print("--"*20)


    # print(np.array(time_arr).shape)
    # print(len(time_arr))

    # source_list = list([8])
    # target_list = list([[1]])

    if args.gpu == "-1":
        save_location = './overhead_results/' + args.outdir[8:-1] + '_cpu_avrg' + str(args.n) + '/'
    else:
        save_location = './overhead_results/' + args.outdir[8:-1] + '_avrg' + str(args.n) + '/'

    directory = save_location

    # Check if the directory exists
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        os.makedirs(directory)

    np.save(save_location + 'time_arr_evaluation_10000.npy', np.array(time_arr))


    with open(save_location + 'acc_and_overhead.txt', 'w') as file:
        file.write(f"Number of parameters: {par}" + '\n')
        file.write("Evaluation time and accuracy\n")
        file.write("beginning_test_acc: " + str(beginning_test_acc) + '\n')
        # file.write("time_all:         " + str(time_all) + '\n')
        file.write("time_arr_all:     " + str(np.array(time_arr).sum()) + '\n')
        # file.write("time_one_avr:     " + str(time_one_avr) + '\n')
        file.write("time_arr_one_avr: " + str(time_arr_avr) + '\n')
        # file.write("time_one:         " + str(time_one) + '\n')
        file.write("--"*20 + '\n')
        file.write("--"*20 + '\n')

        if args.bits == 8:
            t_hd_1 = np.load("tables_hd_bcw/signbit_min_hd_table_C12_3.npy", allow_pickle=True)
            t_hd_2 = np.load("tables_hd_bcw/signbit_min_hd_table_C13_4.npy", allow_pickle=True)
            t_hd_3 = np.load("tables_hd_bcw/signbit_min_hd_table_C14_4.npy", allow_pickle=True)
            t_bcw_1 = np.load("tables_hd_bcw/signbit_tbcw_C12_3.npy", allow_pickle=True)
            t_bcw_2 = np.load("tables_hd_bcw/signbit_tbcw_C13_4.npy", allow_pickle=True)
            t_bcw_3 = np.load("tables_hd_bcw/signbit_tbcw_C14_4.npy", allow_pickle=True)
            t_cw_1 = np.load("tables_hd_bcw/table_code_words_C12.npy", allow_pickle=True)
            t_cw_2 = np.load("tables_hd_bcw/table_code_words_C13.npy", allow_pickle=True)
            t_cw_3 = np.load("tables_hd_bcw/table_code_words_C14.npy", allow_pickle=True)
            t_ec_1 = np.load("tables_hd_bcw/table_error_correction_C12.npy", allow_pickle=True)
            t_ec_2 = np.load("tables_hd_bcw/table_error_correction_C13.npy", allow_pickle=True)
            t_ec_3 = np.load("tables_hd_bcw/table_error_correction_C14.npy", allow_pickle=True)
            arr_table = np.array([[12, t_cw_1, t_ec_1], [13, t_cw_2, t_ec_2], [14, t_cw_3, t_ec_3]], dtype=object)
            max_number = 256
            max_number_2 = 128

        elif args.bits == 4:
            t_hd_1 = np.load("tables_hd_bcw/signbit_min_hd_table_C7_3.npy", allow_pickle=True)
            t_hd_2 = np.load("tables_hd_bcw/signbit_min_hd_table_C8_4.npy", allow_pickle=True)
            t_hd_3 = np.load("tables_hd_bcw/signbit_min_hd_table_C9_4.npy", allow_pickle=True)
            t_bcw_1 = np.load("tables_hd_bcw/signbit_tbcw_C7_3.npy", allow_pickle=True)
            t_bcw_2 = np.load("tables_hd_bcw/signbit_tbcw_C8_4.npy", allow_pickle=True)
            t_bcw_3 = np.load("tables_hd_bcw/signbit_tbcw_C9_4.npy", allow_pickle=True)
            t_cw_1 = np.load("tables_hd_bcw/table_code_words_C7.npy", allow_pickle=True)
            t_cw_2 = np.load("tables_hd_bcw/table_code_words_C8.npy", allow_pickle=True)
            t_cw_3 = np.load("tables_hd_bcw/table_code_words_C9.npy", allow_pickle=True)
            t_ec_1 = np.load("tables_hd_bcw/table_error_correction_C7.npy", allow_pickle=True)
            t_ec_2 = np.load("tables_hd_bcw/table_error_correction_C8.npy", allow_pickle=True)
            t_ec_3 = np.load("tables_hd_bcw/table_error_correction_C9.npy", allow_pickle=True)
            arr_table = np.array([[7, t_cw_1, t_ec_1], [8, t_cw_2, t_ec_2], [9, t_cw_3, t_ec_3]], dtype=object)
            max_number = 16
            max_number_2 = 8


        for code, t_cw, t_ec in arr_table:
            file.write("C" + str(code) + '\n')
            time_decoding_arr = list()

            n_time_encoding = list()
            n_time_decoding = list()
            cpu_encoding = list()
            cpu_decoding = list()

            for n in range(args.n):
                if n % 100 == 0:
                    print("C" + str(code) + "  n: ", n)

                time_decoding_arr_2 = list()
                time_encoding_arr = list()
                cpu_cycles_array_encoding = list()
                cpu_cycles_array_decoding = list()

                # t_cw_v = np.vectorize(lambda x: t_cw[x])
                # t_ec_v = np.vectorize(lambda x: t_ec[x])


                # file.write(str(model.state_dict()))

                weights_get = np.array([], dtype=np.ndarray)
                weights_encoded = np.array([], dtype=np.ndarray)
                weights_decoded = np.array([], dtype=np.ndarray)



                for m in model.cpu().modules():
                    if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):

                        # weights_get = np.append(weights_get, m.weight.data.cpu().numpy())
                        start_time_encoding = time.time()
                        # weight_e = t_cw_v(np.array(m.weight.data.cpu().numpy(), dtype=np.int32) + max_number_2)
                        weight_e = np.vectorize(t_cw.__getitem__)(np.array(m.weight.data.cpu().numpy(), dtype=np.int32) + max_number_2)
                        end_time_encoding = time.time()

                        # start_encoding = count()
                        # weight_e = np.vectorize(t_cw.__getitem__)(np.array(m.weight.data.cpu().numpy(), dtype=np.int32) + max_number_2)
                        # elapsed_encoding = count_end() - start_encoding

                        # weights_encoded = np.append(weights_encoded, np.array(weight_e))

                        # start_time_decoding = time.time()
                        # weight_d = t_ec_v(np.array(weight_e, dtype=np.int32))
                        # end_time_decoding = time.time()

                        start_time_decoding_2 = time.time()
                        weight_d = np.vectorize(t_ec.__getitem__)(np.array(weight_e, dtype=np.int32))
                        end_time_decoding_2 = time.time()

                        # start_decoding = count()
                        # weight_d = np.vectorize(t_ec.__getitem__)(np.array(weight_e, dtype=np.int32))
                        # elapsed_decoding = count_end() - start_decoding

                        # weights_decoded = np.append(weights_decoded, np.array(weight_d))

                        # file.write(str(m.weight.data.cpu().numpy()) + '\n')
                        # file.write("++"*20 + '\n')
                        # file.write(str(weight_e) + '\n')
                        # file.write("##"*20 + '\n')
                        # file.write(str(weight_d) + '\n')
                        # file.write("--"*20 + '\n')

                        time_encoding_arr.append(end_time_encoding - start_time_encoding)
                        # time_decoding_arr.append(end_time_decoding - start_time_decoding)
                        time_decoding_arr_2.append(end_time_decoding_2 - start_time_decoding_2)
                        # cpu_cycles_array_encoding.append(elapsed_encoding)
                        # cpu_cycles_array_decoding.append(elapsed_decoding)

                        # file.write("weights_encoded: \n" + str(weights_encoded) + '\n')

                        # print("time_encoding  : ", np.array(time_encoding_arr).sum())
                        # # print("time_decoding  : ", np.array(time_decoding_arr).sum())
                        # print("time_decoding_2: ", np.array(time_decoding_arr_2).sum())
                        # print("--"*20)

                n_time_encoding.append(np.array(time_encoding_arr).sum())
                n_time_decoding.append(np.array(time_decoding_arr_2).sum())
                # cpu_encoding.append(np.array(cpu_cycles_array_encoding).sum())
                # cpu_decoding.append(np.array(cpu_cycles_array_decoding).sum())

            print("--" * 20)


            file.write("time_encoding_min:  " + str(np.array(n_time_encoding).min()) + '\n')
            file.write("time_decoding_min:  " + str(np.array(n_time_decoding).min()) + '\n')
            file.write("--\n")
            file.write("time_encoding_max:  " + str(np.array(n_time_encoding).max()) + '\n')
            file.write("time_decoding_max:  " + str(np.array(n_time_decoding).max()) + '\n')
            # file.write("time_decoding:   " + str(np.array(time_decoding_arr).sum()) + '\n')
            file.write("--\n")
            file.write("time_encoding_mean: " + str(np.array(n_time_encoding).mean()) + '\n')
            file.write("time_decoding_mean: " + str(np.array(n_time_decoding).mean()) + '\n')

            # file.write("####\n")
            #
            # file.write("cpu_encoding_min: " + str(np.array(cpu_encoding).min()) + '\n')
            # file.write("cpu_decoding_min: " + str(np.array(cpu_decoding).min()) + '\n')
            # file.write("--\n")
            # file.write("cpu_encoding_max: " + str(np.array(cpu_encoding).max()) + '\n')
            # file.write("cpu_decoding_max: " + str(np.array(cpu_decoding).max()) + '\n')
            # file.write("--\n")
            # file.write("cpu_encoding_mean: " + str(np.array(cpu_encoding).mean()) + '\n')
            # file.write("cpu_decoding_mean: " + str(np.array(cpu_decoding).mean()) + '\n')

            np.save(save_location + 'time_arr_encoding.npy', np.array(n_time_encoding))
            # np.save(save_location + 'time_arr_decoding.npy', np.array(time_decoding_arr))
            np.save(save_location + 'time_arr_decoding.npy', np.array(n_time_decoding))
            # np.save(save_location + 'cpu_cycles_array_encoding.npy', np.array(cpu_encoding))
            # np.save(save_location + 'cpu_cycles_array_decoding.npy', np.array(cpu_decoding))
            file.write("--"*20 + '\n')
        file.write(str(model))






if __name__ == "__main__":
    main()