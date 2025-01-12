from __future__ import division
from __future__ import absolute_import

import os, sys, shutil, time
import argparse
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils.attack_bfa_utils.utils_bfa import AverageMeter, time_string, clustering_loss
from tensorboardX import SummaryWriter
import models

from utils.attack_bfa_utils.BFA import *
import copy

import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(
    description='Training network for image classification',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path',
                    # default='/home/elliot/data/pytorch/svhn/',
                    default='data/',
                    type=str,
                    help='Path to dataset')
parser.add_argument(
    '--dataset',
    type=str,
    choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist'],
    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='lbcnn',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs',
                    type=int,
                    # default=200,
                    default=0,
                    help='Number of epochs to train.')
parser.add_argument('--optimizer',
                    type=str,
                    default='SGD',
                    choices=['SGD', 'Adam', 'YF'])
parser.add_argument('--test_batch_size',
                    type=int,
                    default=256,
                    help='Batch size.')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay',
                    type=float,
                    default=1e-4,
                    help='Weight decay (L2 penalty).')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument(
    '--gammas',
    type=float,
    nargs='+',
    default=[0.1, 0.1],
    help=
    'LR is multiplied by gamma on schedule, number of gammas should be equal to schedule'
)
# Checkpoints
parser.add_argument('--print_freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 200)')
parser.add_argument('--save_path',
                    type=str,
                    default='./save/',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument(
    '--fine_tune',
    dest='fine_tune',
    action='store_true',
    help='fine tuning from the pre-trained model, force the start epoch be zero'
)
parser.add_argument('--model_only',
                    dest='model_only',
                    action='store_true',
                    help='only save the model without external utils_')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id',
                    type=int,
                    default=0,
                    help='device range [0,ngpu-1]')
parser.add_argument('--workers',
                    type=int,
                    default=4,
                    help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
# quantization
parser.add_argument(
    '--quan_bitwidth',
    type=int,
    default=None,
    help='the bitwidth used for quantization')
parser.add_argument(
    '--reset_weight',
    dest='reset_weight',
    action='store_true',
    help='enable the weight replacement with the quantized weight')
# Bit Flip Attack
parser.add_argument('--bfa',
                    dest='enable_bfa',
                    action='store_true',
                    help='enable the bit-flip attack_bfa_utils')
parser.add_argument('--attack_sample_size',
                    type=int,
                    default=128,
                    help='attack_bfa_utils sample size')
parser.add_argument('--n_iter',
                    type=int,
                    default=200,
                    help='number of attack_bfa_utils iterations')
parser.add_argument(
    '--k_top',
    type=int,
    default=None,
    help='k weight with top ranking gradient used for bit-level gradient check.'
)
parser.add_argument('--random_bfa',
                    dest='random_bfa',
                    action='store_true',
                    help='perform the bit-flips randomly on weight bits')

# Piecewise clustering
parser.add_argument('--clustering',
                    dest='clustering',
                    action='store_true',
                    help='add the piecewise clustering term.')
parser.add_argument('--lambda_coeff',
                    type=float,
                    default=1e-3,
                    help='lambda coefficient to control the clustering term')

parser.add_argument('--bits', type=int, default=8, help='quantization bits')
parser.add_argument('--outdir', type=str, default='results/', help='folder where the model is saved')
parser.add_argument('--output_act', type=str, default='linear', help='output act. (only linear and tanh is supported)')
parser.add_argument('--gpu', default="0", type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n', default=100, type=int, help='number_of_rounds')
##########################################################################

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        args.gpu_id)  # make only device #gpu_id visible, then

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True


###############################################################################
###############################################################################


def main():
    # Init logger6
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(
        os.path.join(args.save_path,
                     'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')),
              log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()),
              log)

    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log',
                           'run_' + str(args.manualSeed))
    # logger = Logger(tb_path)
    writer = SummaryWriter(tb_path)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'mnist':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])

    if args.dataset == 'mnist':
        train_data = dset.MNIST(args.data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.MNIST(args.data_path,
                               train=False,
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path,
                                   train=True,
                                   transform=train_transform,
                                   download=True)
        test_data = dset.CIFAR100(args.data_path,
                                  train=False,
                                  transform=test_transform,
                                  download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path,
                               split='train',
                               transform=train_transform,
                               download=True)
        test_data = dset.SVHN(args.data_path,
                              split='test',
                              transform=test_transform,
                              download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path,
                                split='train',
                                transform=train_transform,
                                download=True)
        test_data = dset.STL10(args.data_path,
                               split='test',
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'validation')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.attack_sample_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)

    # Init model, criterion, and optimizer
    # net = models_bfa_ori.__dict__[args.arch](num_classes)
    # print_log("=> network :\n {}".format(net), log)
    #
    # if args.use_cuda:
    #     if args.ngpu > 1:
    #         print("1-net")
    #         net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    #         print("2-net")

    # # define loss function (criterion) and optimizer
    # criterion = torch.nn.CrossEntropyLoss()

    # # separate the parameters thus param groups can be updated by different optimizer
    # all_param = [
    #     param for name, param in net.named_parameters()
    #     if not 'step_size' in name
    # ]
    #
    # step_param = [
    #     param for name, param in net.named_parameters() if 'step_size' in name
    # ]
    #
    # if args.optimizer == "SGD":
    #     print("using SGD as optimizer")
    #     optimizer = torch.optim.SGD(all_param,
    #                                 lr=state['learning_rate'],
    #                                 momentum=state['momentum'],
    #                                 weight_decay=state['decay'],
    #                                 nesterov=True)
    #
    # elif args.optimizer == "Adam":
    #     print("using Adam as optimizer")
    #     optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad,
    #                                         all_param),
    #                                  lr=state['learning_rate'],
    #                                  weight_decay=state['decay'])
    #
    # elif args.optimizer == "RMSprop":
    #     print("using RMSprop as optimizer")
    #     optimizer = torch.optim.RMSprop(
    #         filter(lambda param: param.requires_grad, net.parameters()),
    #         lr=state['learning_rate'],
    #         alpha=0.99,
    #         eps=1e-08,
    #         weight_decay=0,
    #         momentum=0)
    #
    # if args.use_cuda:
    #     net.cuda()
    #     criterion.cuda()
    #
    # recorder = RecorderMeter(args.epochs)  # count number of epoches
    #
    # # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print_log("=> loading checkpoint '{}'".format(args.resume), log)
    #         checkpoint = torch.load(args.resume)
    #         if not (args.fine_tune):
    #             args.start_epoch = checkpoint['epoch']
    #             recorder = checkpoint['recorder']
    #             optimizer.load_state_dict(checkpoint['optimizer'])
    #
    #         state_tmp = net.state_dict()
    #         if 'state_dict' in checkpoint.keys():
    #             state_tmp.update(checkpoint['state_dict'])
    #         else:
    #             state_tmp.update(checkpoint)
    #
    #         net.load_state_dict(state_tmp)
    #
    #         print_log(
    #             "=> loaded checkpoint '{}' (epoch {})".format(
    #                 args.resume, args.start_epoch), log)
    #     else:
    #         print_log("=> no checkpoint found at '{}'".format(args.resume),
    #                   log)
    # else:
    #     print_log(
    #         "=> do not use any checkpoint for {} model".format(args.arch), log)
    #
    # # Configure the quantization bit-width
    # if args.quan_bitwidth is not None:
    #     change_quan_bitwidth(net, args.quan_bitwidth)
    #
    # # update the step_size once the model is loaded. This is used for quantization.
    # for m in net.modules():
    #     if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
    #         # simple step size update based on the pretrained model or weight init
    #         m.__reset_stepsize__()
    #
    # # block for weight reset
    # if args.reset_weight:
    #     for m in net.modules():
    #         if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
    #             m.__reset_weight__()
    #             # print(m.weight)

    device = torch.device('cuda')
    gpu_list = [int(i) for i in args.gpu.strip().split(",")] if args.gpu is not "0" else [0]
    number_of_rounds = args.n
    weights = np.empty((number_of_rounds, 2), dtype=np.ndarray)
    weights_all = np.empty((1, 2), dtype=np.ndarray)

    save_location = './bfa_results/' + args.outdir[8:-1] + "_" + str(number_of_rounds) + "/"

    directory = save_location

    # Check if the directory exists
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        os.makedirs(directory)

    criterion = torch.nn.CrossEntropyLoss()
    if args.use_cuda:
        criterion.cuda()

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        model = models.__dict__[args.arch](num_classes, args.bits, args.output_act)
    elif args.dataset == 'imagenet':
        model, state_dict = models.__dict__[args.arch](num_classes, args.bits)

    # model = models.__dict__[args.arch](num_classes, args.bits, args.output_act)
    model = torch.nn.DataParallel(model, gpu_list).to(device) if len(gpu_list) > 1 else torch.nn.DataParallel(model).to(
        device)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        state_dict = torch.load(args.outdir + 'model_best.pth.tar', map_location=device)['state_dict']
        model.load_state_dict(state_dict)

    print(model)
    attacker = BFA(criterion, model, args.k_top)
    # model_clean = copy.deepcopy(model)
    # weight_conversion_2(model_clean)
    weight_conversion_2(model)

    if args.enable_bfa:
        start_time = time.time()

        for i in range(number_of_rounds):
            model.load_state_dict(state_dict)
            weight_conversion_2(model)

            weights_all[0][0] = get_all_weights(model)

            model = model.cuda()
            model.eval()

            perform_attack(attacker, model, train_loader, test_loader, args.n_iter, log, writer,
                           csv_save_path=args.save_path, random_attack=args.random_bfa)

            # perform_attack(attacker, model, model_clean, train_loader, test_loader, args.n_iter, log, writer,
            #                csv_save_path=args.save_path, random_attack=args.random_bfa)


            weights_all[0][1] = get_all_weights(model)

            dif = compare_weights(weights_all)

            weights[i][0] = dif[0][0]
            weights[i][1] = dif[0][1]

            try:
                np.save(save_location + 'weights.npy', weights)
            except:

                directory = save_location + 'weights/'

                # Check if the directory exists
                if not os.path.exists(directory):
                    # If it doesn't exist, create it
                    os.makedirs(directory)

                i = 0
                for weight in weights:
                    np.save(save_location + 'weights/weights_' + str(i) + '.npy', weight)
                    i += 1

            print(
                "#############################################################################################################")
            print(str(i + 1))
            print(str(time.time() - start_time))
            print(
                "#############################################################################################################")

        end_time = time.time()



        with open(save_location + 'evaluation.txt', 'w') as file:
            # file.write('hello world !')

            # print("-------------------")
            # print(weights.shape)

            # print(weights)

            count = np.zeros(weights.shape[0])

            hd_0 = np.zeros([weights.shape[0]])
            hd_1 = np.zeros([weights.shape[0]])
            hd_2 = np.zeros([weights.shape[0]])
            hd_3 = np.zeros([weights.shape[0]])

            if args.bits == 8:
                t_hd_1 = np.load("./tables_hd_bcw/signbit_min_hd_table_C12_3.npy", allow_pickle=True)
                t_hd_2 = np.load("./tables_hd_bcw/signbit_min_hd_table_C13_4.npy", allow_pickle=True)
                t_hd_3 = np.load("./tables_hd_bcw/signbit_min_hd_table_C14_4.npy", allow_pickle=True)
                t_cw_1 = np.load("./tables_hd_bcw/signbit_tbcw_C12_3.npy", allow_pickle=True)
                t_cw_2 = np.load("./tables_hd_bcw/signbit_tbcw_C13_4.npy", allow_pickle=True)
                t_cw_3 = np.load("./tables_hd_bcw/signbit_tbcw_C14_4.npy", allow_pickle=True)
                max_number = 256
                max_number_2 = 128
            elif args.bits == 4:
                t_hd_1 = np.load("./tables_hd_bcw/signbit_min_hd_table_C7_3.npy", allow_pickle=True)
                t_hd_2 = np.load("./tables_hd_bcw/signbit_min_hd_table_C8_4.npy", allow_pickle=True)
                t_hd_3 = np.load("./tables_hd_bcw/signbit_min_hd_table_C9_4.npy", allow_pickle=True)
                t_cw_1 = np.load("./tables_hd_bcw/signbit_tbcw_C7_3.npy", allow_pickle=True)
                t_cw_2 = np.load("./tables_hd_bcw/signbit_tbcw_C8_4.npy", allow_pickle=True)
                t_cw_3 = np.load("./tables_hd_bcw/signbit_tbcw_C9_4.npy", allow_pickle=True)
                max_number = 16
                max_number_2 = 8

            for i in range(weights.shape[0]):
                # print(weights[i][0].shape)
                # print(weights[i][1].shape)
                # print(weights[i][0])
                # print(weights[i][1])

                file.write('run number: ' + str(i) + '\n')
                file.write(str(weights[i][0].shape) + '\n')
                file.write(str(weights[i][1].shape) + '\n')
                file.write('-------------------' + '\n')

                for z, y in zip(weights[i][0], weights[i][1]):
                    if z != y:
                        count[i] += 1
                        # print(z, y)
                        # print("-------------------")

                        if z < 0:
                            z_1 = max_number + z
                        else:
                            z_1 = z
                        if y < 0:
                            y_1 = max_number + y
                        else:
                            y_1 = y

                        bin_z = [int(bit_z) for bit_z in bin(int(z_1))[2:].zfill(args.bits)]
                        bin_y = [int(bit_y) for bit_y in bin(int(y_1))[2:].zfill(args.bits)]

                        hd_0[i] += hamming(bin_z, bin_y) * len(bin_z)
                        hd_1[i] += t_hd_1[int(z) + max_number_2][int(y) + max_number_2]
                        hd_2[i] += t_hd_2[int(z) + max_number_2][int(y) + max_number_2]
                        hd_3[i] += t_hd_3[int(z) + max_number_2][int(y) + max_number_2]

                        file.write(str(z) + ' ' + str(y) + '\n' + str(bin_z) + '\n' + str(
                            bin_y) + '\nhamming distance: ' + str(hamming(bin_z, bin_y) * len(bin_z)) + '\n\n')
                        file.write(str(t_cw_1[int(z) + max_number_2]) + '\n' + str(
                            t_cw_1[int(y) + max_number_2]) + '\nhamming distance: ' + str(
                            t_hd_1[int(z) + max_number_2][int(y) + max_number_2]) + '\n\n')
                        file.write(str(t_cw_2[int(z) + max_number_2]) + '\n' + str(
                            t_cw_2[int(y) + max_number_2]) + '\nhamming distance: ' + str(
                            t_hd_2[int(z) + max_number_2][int(y) + max_number_2]) + '\n\n')
                        file.write(str(t_cw_3[int(z) + max_number_2]) + '\n' + str(
                            t_cw_3[int(y) + max_number_2]) + '\nhamming distance: ' + str(
                            t_hd_3[int(z) + max_number_2][int(y) + max_number_2]) + '\n')
                        file.write("-------------------" + '\n')

                # print('number of weights attacked: ', count[i])
                # print("#################")
                # print("#################")

                file.write('number of weights attacked: ' + str(count[i]) + '\n')
                file.write('hd_0: ' + str(hd_0[i]) + '\n')
                file.write('hd_1: ' + str(hd_1[i]) + '\n')
                file.write('hd_2: ' + str(hd_2[i]) + '\n')
                file.write('hd_3: ' + str(hd_3[i]) + '\n')
                file.write("#################" + '\n')
                file.write("#################" + '\n')
                file.write('\n\n\n')

            hd_0_mean = np.mean(hd_0)
            hd_1_mean = np.mean(hd_1)
            hd_2_mean = np.mean(hd_2)
            hd_3_mean = np.mean(hd_3)

            hd_0_std = np.std(hd_0)
            hd_1_std = np.std(hd_1)
            hd_2_std = np.std(hd_2)
            hd_3_std = np.std(hd_3)

            print('Time:', end_time - start_time, 's')
            file.write('Time_of_execution: ' + str(end_time - start_time) + 's' + '\n')

            file.write('# of weights attacked mean: ' + str(count.mean()) + '\n')
            file.write('# of weights attacked std: ' + str(count.std()) + '\n')
            file.write('hd_0_mean: ' + str(hd_0_mean) + '\n')
            file.write('hd_0_std: ' + str(hd_0_std) + '\n')
            file.write('hd_1_mean: ' + str(hd_1_mean) + '\n')
            file.write('hd_1_std: ' + str(hd_1_std) + '\n')
            file.write('hd_2_mean: ' + str(hd_2_mean) + '\n')
            file.write('hd_2_std: ' + str(hd_2_std) + '\n')
            file.write('hd_3_mean: ' + str(hd_3_mean) + '\n')
            file.write('hd_3_std: ' + str(hd_3_std) + '\n')

        try:
            np.save(save_location + 'weights.npy', weights)
        except:

            directory = save_location + 'weights/'

            # Check if the directory exists
            if not os.path.exists(directory):
                # If it doesn't exist, create it
                os.makedirs(directory)

            i = 0
            for weight in weights:
                np.save(save_location + 'weights/weights_' + str(i) + '.npy', weight)
                i += 1

        np.save(save_location + 'numbers_of_weights_attacked.npy', count)

        return

    if args.evaluate:
        _, _, _, output_summary = validate(test_loader, model, criterion, log, summary_output=True)
        pd.DataFrame(output_summary).to_csv(os.path.join(args.save_path, 'output_summary_{}.csv'.format(args.arch)),
                                            header=['top-1 output'], index=False)

        # print(top1, top5, loss)

        return

    # # Main loop
    # start_time = time.time()
    # epoch_time = AverageMeter()
    #
    # for epoch in range(args.start_epoch, args.epochs):
    #     current_learning_rate, current_momentum = adjust_learning_rate(
    #         optimizer, epoch, args.gammas, args.schedule)
    #     # Display simulation time
    #     need_hour, need_mins, need_secs = convert_secs2time(
    #         epoch_time.avg * (args.epochs - epoch))
    #     need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
    #         need_hour, need_mins, need_secs)
    #
    #     print_log(
    #         '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}][M={:1.2f}]'.format(time_string(), epoch, args.epochs,
    #                                                                                need_time, current_learning_rate,
    #                                                                                current_momentum) \
    #         + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
    #                                                            100 - recorder.max_accuracy(False)), log)
    #
    #     # train for one epoch
    #     train_acc, train_los = train(train_loader, net, criterion, optimizer,
    #                                  epoch, log)
    #
    #     # evaluate on validation set
    #     val_acc, _, val_los = validate(test_loader, net, criterion, log)
    #     recorder.update(epoch, train_los, train_acc, val_los, val_acc)
    #     is_best = val_acc >= recorder.max_accuracy(False)
    #
    #     if args.model_only:
    #         checkpoint_state = {'state_dict': net.state_dict}
    #     else:
    #         checkpoint_state = {
    #             'epoch': epoch + 1,
    #             'arch': args.arch,
    #             'state_dict': net.state_dict(),
    #             'recorder': recorder,
    #             'optimizer': optimizer.state_dict(),
    #         }
    #
    #     save_checkpoint(checkpoint_state, is_best, args.save_path,
    #                     'checkpoint.pth.tar', log)
    #
    #     # measure elapsed time
    #     epoch_time.update(time.time() - start_time)
    #     start_time = time.time()
    #     recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))
    #
    #     # save addition accuracy log for plotting
    #     accuracy_logger(base_dir=args.save_path,
    #                     epoch=epoch,
    #                     train_accuracy=train_acc,
    #                     test_accuracy=val_acc)
    #
    #     # ============ TensorBoard logging ============#
    #
    #     ## Log the graidents distribution
    #     for name, param in net.named_parameters():
    #         name = name.replace('.', '/')
    #         try:
    #             writer.add_histogram(name + '/grad',
    #                                 param.grad.clone().cpu().data.numpy(),
    #                                 epoch + 1,
    #                                 bins='tensorflow')
    #         except:
    #             pass
    #
    #         try:
    #             writer.add_histogram(name, param.clone().cpu().data.numpy(),
    #                                   epoch + 1, bins='tensorflow')
    #         except:
    #             pass
    #
    #     total_weight_change = 0
    #
    #     for name, module in net.named_modules():
    #         if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
    #             try:
    #                 writer.add_histogram(name+'/bin_weight', module.bin_weight.clone().cpu().data.numpy(), epoch + 1,
    #                                     bins='tensorflow')
    #                 writer.add_scalar(name + '/bin_weight_change', module.bin_weight_change, epoch+1)
    #                 total_weight_change += module.bin_weight_change
    #                 writer.add_scalar(name + '/bin_weight_change_ratio', module.bin_weight_change_ratio, epoch+1)
    #             except:
    #                 pass
    #
    #     writer.add_scalar('total_weight_change', total_weight_change, epoch + 1)
    #     print('total weight changes:', total_weight_change)
    #
    #     writer.add_scalar('loss/train_loss', train_los, epoch + 1)
    #     writer.add_scalar('loss/test_loss', val_los, epoch + 1)
    #     writer.add_scalar('accuracy/train_accuracy', train_acc, epoch + 1)
    #     writer.add_scalar('accuracy/test_accuracy', val_acc, epoch + 1)
    # # ============ TensorBoard logging ============#
    #
    log.close()


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
def perform_attack(attacker, model, train_loader, test_loader,
                   N_iter, log, writer, csv_save_path=None, random_attack=False):

# def perform_attack(attacker, model, model_clean, train_loader, test_loader,
#                    N_iter, log, writer, csv_save_path=None, random_attack=False):

    # Note that, attack_bfa_utils has to be done in evaluation model due to batch-norm.
    # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
    model.eval()
    losses = AverageMeter()
    iter_time = AverageMeter()
    attack_time = AverageMeter()

    # attempt to use the training data to conduct BFA
    for _, (data, target) in enumerate(train_loader):
        if args.use_cuda:
            target = target.cuda(non_blocking=True)
            data = data.cuda()
        # Override the target to prevent label leaking
        _, target = model(data).data.max(1)
        break

    # evaluate the test accuracy of clean model
    val_acc_top1, val_acc_top5, val_loss, output_summary = validate(test_loader, model,
                                                                    attacker.criterion, log, summary_output=True)
    tmp_df = pd.DataFrame(output_summary, columns=['top-1 output'])
    tmp_df['BFA iteration'] = 0
    tmp_df.to_csv(os.path.join(args.save_path, 'output_summary_{}_BFA_0.csv'.format(args.arch)),
                  index=False)

    writer.add_scalar('attack_bfa_utils/val_top1_acc', val_acc_top1, 0)
    writer.add_scalar('attack_bfa_utils/val_top5_acc', val_acc_top5, 0)
    writer.add_scalar('attack_bfa_utils/val_loss', val_loss, 0)

    print_log('k_top is set to {}'.format(args.k_top), log)
    print_log('Attack sample size is {}'.format(data.size()[0]), log)
    end = time.time()

    df = pd.DataFrame()  # init a empty dataframe for logging
    last_val_acc_top1 = val_acc_top1

    for i_iter in range(N_iter):
        print_log('**********************************', log)
        if not random_attack:
            attack_log = attacker.progressive_bit_search(model, data, target)
        else:
            attack_log = attacker.random_flip_one_bit(model)

        # measure data loading time
        attack_time.update(time.time() - end)
        end = time.time()

        # h_dist = hamming_distance(model, model_clean)

        # record the loss
        if hasattr(attacker, "loss_max"):
            losses.update(attacker.loss_max, data.size(0))

        print_log(
            'Iteration: [{:03d}/{:03d}]   '
            'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
            format((i_iter + 1),
                   N_iter,
                   attack_time=attack_time,
                   iter_time=iter_time) + time_string(), log)
        try:
            print_log('loss before attack_bfa_utils: {:.4f}'.format(attacker.loss.item()),
                      log)
            print_log('loss after attack_bfa_utils: {:.4f}'.format(attacker.loss_max), log)
        except:
            pass

        print_log('bit flips: {:.0f}'.format(attacker.bit_counter), log)
        # print_log('hamming_dist: {:.0f}'.format(h_dist), log)

        writer.add_scalar('attack_bfa_utils/bit_flip', attacker.bit_counter, i_iter + 1)
        # writer.add_scalar('attack_bfa_utils/h_dist', h_dist, i_iter + 1)
        writer.add_scalar('attack_bfa_utils/sample_loss', losses.avg, i_iter + 1)

        # exam the BFA on entire val dataset
        val_acc_top1, val_acc_top5, val_loss, output_summary = validate(
            test_loader, model, attacker.criterion, log, summary_output=True)
        tmp_df = pd.DataFrame(output_summary, columns=['top-1 output'])
        tmp_df['BFA iteration'] = i_iter + 1
        tmp_df.to_csv(os.path.join(args.save_path, 'output_summary_{}_BFA_{}.csv'.format(args.arch, i_iter + 1)),
                      index=False)

        # add additional info for logging
        acc_drop = last_val_acc_top1 - val_acc_top1
        last_val_acc_top1 = val_acc_top1

        # print(attack_log)
        for i in range(attack_log.__len__()):
            attack_log[i].append(val_acc_top1)
            attack_log[i].append(acc_drop)
        # print(attack_log)
        df = df.append(attack_log, ignore_index=True)

        writer.add_scalar('attack_bfa_utils/val_top1_acc', val_acc_top1, i_iter + 1)
        writer.add_scalar('attack_bfa_utils/val_top5_acc', val_acc_top5, i_iter + 1)
        writer.add_scalar('attack_bfa_utils/val_loss', val_loss, i_iter + 1)

        # measure elapsed time
        iter_time.update(time.time() - end)
        print_log(
            'iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(
                iter_time=iter_time), log)
        end = time.time()

        # Stop the attack_bfa_utils if the accuracy is below the configured break_acc.
        if args.dataset == 'cifar10':
            break_acc = 11.0
        elif args.dataset == 'cifar100':
            break_acc = 1.2
        elif args.dataset == 'imagenet':
            break_acc = 0.2
        if val_acc_top1 <= break_acc:
            break

    # attack_bfa_utils profile
    column_list = ['module idx', 'bit-flip idx', 'module name', 'weight idx',
                   'weight before attack_bfa_utils', 'weight after attack_bfa_utils', 'validation accuracy',
                   'accuracy drop']
    df.columns = column_list
    df['trial seed'] = args.manualSeed
    if csv_save_path is not None:
        csv_file_name = 'attack_profile_{}.csv'.format(args.manualSeed)
        export_csv = df.to_csv(os.path.join(csv_save_path, csv_file_name), index=None)

    return


# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(
                non_blocking=True
            )  # the copy will be asynchronous with respect to the host.
            input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)
        if args.clustering:
            loss += clustering_loss(model, args.lambda_coeff)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log(
                '  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
        .format(top1=top1, top5=top5, error1=100 - top1.avg), log)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log, summary_output=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    output_summary = []  # init a list for output summary

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda(non_blocking=True)
                input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # summary the output
            if summary_output:
                tmp_list = output.max(1, keepdim=True)[
                    1].flatten().cpu().numpy()  # get the index of the max log-probability
                output_summary.append(tmp_list)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

        print_log(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
            .format(top1=top1, top5=top5, error1=100 - top1.avg), log)

    if summary_output:
        output_summary = np.asarray(output_summary).flatten()
        return top1.avg, top5.avg, losses.avg, output_summary
    else:
        return top1.avg, top5.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write(
            '{epoch}       {train}    {test}\n'.format(**recorder))


if __name__ == '__main__':
    main()
