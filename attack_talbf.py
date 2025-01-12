import argparse
import copy
import multiprocessing
import time

from bitstring import Bits
import datasets
import models
from utils import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, L1Loss

from scipy.spatial.distance import hamming
import concurrent.futures
## This script is adapted from the following public repository:
## https://github.com/jiawangbai/TA-LBF

parser = argparse.ArgumentParser(description='Stealthy TA-LBF on DNNs')
parser.add_argument('--tc', type=str, default=0, help='target class')
parser.add_argument('--data_dir', type=str, default='data/')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset for processing')
parser.add_argument('--num_classes', '-c', default=10, type=int, help='number of classes in the dataset')
parser.add_argument('--arch', '-a', type=str, default='resnet20_quan', help='model architecture')
parser.add_argument('--bits', type=int, default=8, help='quantization bits')
parser.add_argument('--ocm', action='store_true', help='output layer coding with bit strings')
parser.add_argument('--output_act', type=str, default='linear', help='output act. (only linear and tanh is supported)')
parser.add_argument('--code_length', '-cl', default=16, type=int, help='length of codewords')
parser.add_argument('--outdir', type=str, default='results/cifar10/resnet20_quan8_OCM64/', help='folder where the model is saved')
parser.add_argument('--batch', '-b', default=128, type=int, metavar='N', help='Mini-batch size (default: 128)')
parser.add_argument('--gpu', default="0", type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--attack_info', type=str, default='cifar10_talbf.txt', help='attack info list')
parser.add_argument('--init-k', '-init_k', default=5, type=float)
parser.add_argument('--init-lam', '-init_lam', default=100, type=float)
parser.add_argument('--max-search-k', '-max_search_k', default=6, type=int)
parser.add_argument('--max-search-lam', '-max_search_lam', default=8, type=int)
parser.add_argument('--n_aux', type=int, default=64, help='number of auxiliary samples')
parser.add_argument('--initial-rho1', '-initial_rho1', default=0.0001, type=float)
parser.add_argument('--initial-rho2', '-initial_rho2', default=0.0001, type=float)
parser.add_argument('--initial-rho3', '-initial_rho3', default=0.00001, type=float)
parser.add_argument('--max-rho1', '-max_rho1', default=50, type=float)
parser.add_argument('--max-rho2', '-max_rho2', default=50, type=float)
parser.add_argument('--max-rho3', '-max_rho3', default=5, type=float)
parser.add_argument('--rho-fact', '-rho_fact', default=1.01, type=float)
parser.add_argument('--inn-lr', '-inn_lr', default=0.01, type=float)
parser.add_argument('--ext-max-iters', '-ext_max_iters', default=2000, type=int)
parser.add_argument('--inn-max-iters', '-inn_max_iters', default=5, type=int)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--n', default=100, type=int, help='number_of_rounds')
args = parser.parse_args()

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(args.seed)
torch.manual_seed(args.seed)
gpu_list = [int(i) for i in args.gpu.strip().split(",")] if args.gpu is not "0" else [0]
if args.gpu == "-1":
    device = torch.device('cpu')
    print('Using cpu')
else:
    device = torch.device('cuda')
    print('Using gpu: ' + args.gpu)


class AugLag(nn.Module):
    def __init__(self, n_bits, w, b, step_size, args, C):
        super(AugLag, self).__init__()
        self.n_bits = n_bits
        self.b = nn.Parameter(torch.tensor(b).float(), requires_grad=True)
        self.w_twos = nn.Parameter(torch.zeros([w.shape[0], w.shape[1], self.n_bits]), requires_grad=True)
        self.step_size = step_size
        self.w = w
        self.args = args
        self.C = C
        self.output_act = nn.Tanh() if args.output_act == 'tanh' else None

        self.reset_w_twos()
        base = [2**i for i in range(self.n_bits-1, -1, -1)]
        base[0] = -base[0]
        self.base = nn.Parameter(torch.tensor([[base]]).float())

    def forward(self, x):
        w = self.w_twos * self.base
        w = torch.sum(w, dim=2) * self.step_size
        x = F.linear(x, w, self.b)
        if self.args.ocm:
            x = nn.Sigmoid()(2 * x)         # scale to [0, 1]
        else:
            x = self.output_act(x) if self.output_act is not None else x
        return x

    def predict(self, x):
        x = self.forward(x)
        x = 2 * x - 1 if self.args.ocm else x       # rescale OCM output back to [-1, 1] for our usual way of prediction
        x = F.softmax(torch.log(F.relu(torch.matmul(x, self.C.T)) + 1e-6)) if self.args.ocm else F.softmax(x)
        return x

    def reset_w_twos(self):
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                self.w_twos.data[i][j] += torch.tensor([int(b) for b in Bits(int=int(self.w[i][j]), length=self.n_bits).bin])

    # # function to convert binary weights to int weights
    # def convert_w_twos(self):
    #     for i in range(self.w.shape[0]):
    #         for j in range(self.w.shape[1]):
    #             self.w_twos.data[i][j] = torch.tensor([int(b) for b in Bits(bin=self.w_twos.data[i][j].detach().cpu().numpy()).int])


def project_box(x):
    xp = x
    xp[x > 1] = 1
    xp[x < 0] = 0
    return xp


def project_shifted_Lp_ball(x, p):
    shift_vec = 1/2*np.ones(x.size)
    shift_x = x-shift_vec
    normp_shift = np.linalg.norm(shift_x, p)
    n = x.size
    xp = (n**(1/p)) * shift_x / (2*normp_shift) + shift_vec
    return xp


def project_positive(x):
    xp = np.clip(x, 0, None)
    return xp


def loss_func(output, labels, s, t, lam, w_twos, b_ori, k_bits, y1, y2, y3, z1, z2, z3, rho1, rho2, rho3, C):
    if args.ocm:
        # applying the tanh to sigmoid trick to be able to compute BCELoss and eventually attack OCM models via TA-LBF
        C_shift = (C + 1) / 2
        output = torch.nan_to_num(output)
        l1 = torch.nn.BCELoss()(output[-1], C_shift[t])
        l2 = torch.nn.BCELoss()(output[:-1], C_shift[labels[:-1]])
    else:
        l1 = - torch.log(torch.nn.Softmax()(output[-1]))[t]
        l2 = CrossEntropyLoss()(output[:-1], labels[:-1])
    b_ori = torch.tensor(b_ori).float().cuda()
    b = w_twos.view(-1)

    y1, y2, y3, z1, z2, z3 = torch.tensor(y1).float().cuda(), torch.tensor(y2).float().cuda(), \
                             torch.tensor(y3).float().cuda(), torch.tensor(z1).float().cuda(), \
                             torch.tensor(z2).float().cuda(), torch.tensor(z3).float().cuda()

    l3 = z1@(b-y1) + z2@(b-y2) + z3*(torch.norm(b - b_ori) ** 2 - k_bits + y3)

    l4 = (rho1/2) * torch.norm(b - y1) ** 2 + (rho2/2) * torch.norm(b - y2) ** 2 + \
         (rho3/2) * (torch.norm(b - b_ori)**2 - k_bits + y3) ** 2

    return l1 + lam * l2 + l3 + l4


def attack(auglag_ori, all_data, labels, labels_cuda, target_idx, target_class, source_class, aux_idx, lam, k_bits, args):
    n_aux = args.n_aux
    lam = lam
    ext_max_iters = args.ext_max_iters
    inn_max_iters = args.inn_max_iters
    initial_rho1 = args.initial_rho1
    initial_rho2 = args.initial_rho2
    initial_rho3 = args.initial_rho3
    max_rho1 = args.max_rho1
    max_rho2 = args.max_rho2
    max_rho3 = args.max_rho3
    rho_fact = args.rho_fact
    inn_lr = args.inn_lr

    all_idx = np.append(aux_idx, target_idx)
    auglag = copy.deepcopy(auglag_ori)

    b_ori = auglag.w_twos.data.view(-1).detach().cpu().numpy()
    b_new = b_ori

    y1, y2, y3 = b_ori, b_ori, 0
    z1, z2, z3 = np.zeros_like(y1), np.zeros_like(y1), 0
    rho1, rho2, rho3 = initial_rho1, initial_rho2, initial_rho3

    stop_flag = False
    for ext_iter in range(ext_max_iters):
        y1 = project_box(b_new + z1 / rho1)
        y2 = project_shifted_Lp_ball(b_new + z2 / rho2, p=2)
        y3 = project_positive(-np.linalg.norm(b_new - b_ori, ord=2) ** 2 + k_bits - z3 / rho3)

        for inn_iter in range(inn_max_iters):
            input_var = torch.autograd.Variable(all_data[all_idx], volatile=True)
            target_var = torch.autograd.Variable(labels_cuda[all_idx].long(), volatile=True)

            output = auglag(input_var)
            loss = loss_func(output, target_var, source_class, target_class, lam, auglag.w_twos,
                             b_ori, k_bits, y1, y2, y3, z1, z2, z3, rho1, rho2, rho3, auglag.C)

            loss.backward(retain_graph=True)
            auglag.w_twos.data = auglag.w_twos.data - inn_lr * auglag.w_twos.grad.data
            auglag.w_twos.grad.zero_()

        b_new = auglag.w_twos.data.view(-1).detach().cpu().numpy()

        # flag_isnan = 0
        if True in np.isnan(b_new):
            # flag_isnan = -1
            return -1

        z1 = z1 + rho1 * (b_new - y1)
        z2 = z2 + rho2 * (b_new - y2)
        z3 = z3 + rho3 * (np.linalg.norm(b_new - b_ori, ord=2) ** 2 - k_bits + y3)

        rho1 = min(rho_fact * rho1, max_rho1)
        rho2 = min(rho_fact * rho2, max_rho2)
        rho3 = min(rho_fact * rho3, max_rho3)

        temp1 = (np.linalg.norm(b_new - y1)) / max(np.linalg.norm(b_new), 2.2204e-16)
        temp2 = (np.linalg.norm(b_new - y2)) / max(np.linalg.norm(b_new), 2.2204e-16)

        if max(temp1, temp2) <= 1e-4 and ext_iter > 100:
            print('END iter: %d, stop_threshold: %.6f, loss: %.4f' % (ext_iter, max(temp1, temp2), loss.item()))
            stop_flag = True
            break

    auglag.w_twos.data[auglag.w_twos.data > 0.5] = 1.0
    auglag.w_twos.data[auglag.w_twos.data < 0.5] = 0.0

    output = auglag.predict(all_data)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze(1)

    expose_list = [i for i in range(len(output)) if labels[i].to('cpu') == pred[i].to('cpu') and i != target_idx and i not in aux_idx]
    pa_acc = len(expose_list) / (len(labels) - 1 - n_aux)
    n_bit = torch.norm(auglag_ori.w_twos.data.view(-1) - auglag.w_twos.data.view(-1), p=0).item()
    ret = {"pa_acc": pa_acc, "stop": stop_flag, "suc": target_class == pred[target_idx].item(), "n_bit": n_bit, "auglag_new": auglag}

    return ret


def load_data(model, test_loader, args, C):
    mid_out, labels = np.zeros([len(test_loader.dataset), model.mid_dim]), np.zeros([len(test_loader.dataset)])
    start = 0
    model.eval()
    # print(labels.shape)
    print(test_loader.dataset.__len__())
    for i, (input, target) in enumerate(test_loader):
        if i % 100 == 0:
            print("round enumarate test_loader",i)
        # print("round enumarate test_loader", i)
        if C is not None:
            target = torch.tensor([torch.where(torch.all(C.to('cpu') == target[i], dim=1))[0][0] for i in range(target.shape[0])])
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        output = model(input_var)
        mid_out[start: start + args.batch] = output.detach().cpu().numpy()
        labels[start: start + args.batch] = target.numpy()
        start += args.batch
    mid_out = torch.tensor(mid_out).float().cuda()
    labels = torch.tensor(labels).float()
    return mid_out, labels


def load_model(args, DATASET):
    n_output = args.code_length if args.ocm else args.num_classes
    C = torch.tensor(DATASET.C).to(device) if args.ocm else None

    # Evaluate clean accuracy
    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        print('loading model from ' + args.outdir + 'model_best.pth.tar')
        model = models.__dict__[args.arch + '_mid'](n_output, args.bits)
    elif args.dataset == 'ImageNet':
        model, state_dict = models.__dict__[args.arch + '_mid'](n_output, args.bits)

    print(model)
    model = nn.DataParallel(model, gpu_list).to(device) if len(gpu_list) > 1 else nn.DataParallel(model).to(device)
    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        state_dict = torch.load(args.outdir + 'model_best.pth.tar', map_location=device)['state_dict']
        model.load_state_dict(state_dict)
        # print(torch.load(args.outdir + 'model_best.pth.tar'))
    weight_conversion(model)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    weight = model.linear.weight.data.detach().cpu().numpy()
    bias = model.linear.bias.data.detach().cpu().numpy()
    step_size = np.float32(model.linear.step_size.detach().cpu().numpy())

    return weight, bias, step_size, model, C, state_dict

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

def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val                         # return positive value as is


def main():
    # Load dataset
    DATASET = datasets.__dict__[args.dataset](args)
    _, test_loader = DATASET.loaders()

    weight, bias, step_size, model, C, state_dict = load_model(args, DATASET)
    mid_out, labels = load_data(model, test_loader, args, C)
    labels_cuda = labels.cuda()
    
    auglag = AugLag(args.bits, weight, bias, step_size, args, C).cuda()
    clean_output = auglag.predict(mid_out)
    _, pred = clean_output.cpu().topk(1, 1, True, True)
    corrects = [i for i in range(len(pred.squeeze(1))) if labels[i] == pred.squeeze(1)[i]]
    acc_ori = len([i for i in range(len(pred.squeeze(1))) if labels[i] == pred.squeeze(1)[i]]) / len(labels)



    # with open('imagenet_resnet18_w_' + str(args.bits) + '.txt', 'w') as f:
    #     for i, w in enumerate(get_all_weights(model)):
    #         f.write(str(w) + ',')
    #         if i%100 == 0:
    #             f.write('\n')

    print('Original ACC: ', acc_ori)



    n_attacks = args.n
    start_time = time.time()
    weights = np.empty((n_attacks, 2), dtype=np.ndarray)
    weights_all = np.empty((1, 2), dtype=np.ndarray)

    save_location = './talbf_results/' + args.outdir[8:-1] + "_" + str(n_attacks) + "_targetClass" + str(args.tc) + '/'

    directory = save_location

    # Check if the directory exists
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        os.makedirs(directory)

    print("Attack Start")
    attack_info = np.loadtxt(args.attack_info).astype(int)

    # if args.dataset == 'CIFAR10':
    #     attack_info = np.loadtxt(args.attack_info).astype(int)
    # if args.dataset == 'ImageNet':
    #     print(labels.shape)
    #     attack_idx = np.random.randint(labels.shape[0], size=1000)
    #     target_class = np.random.randint(1000, size=1000)
    #
    #     for i in range(1000):
    #         while target_class[i] == labels[attack_idx[i]]:
    #             target_class[i] = np.random.randint(1000)
    #             print('change of target class')
    #
    #     with open('imagenet_talbf.txt', 'w') as f:
    #         for i in range(1000):
    #             f.write(str(target_class[i]) + ' ' + str(attack_idx[i]) + '\n')
    #
    #     print('attack info saved')
    #     attack_info = np.loadtxt(args.attack_info).astype(int)
    # if args.dataset == 'CIFAR100':
    #     print(labels.shape)
    #     attack_idx = np.random.randint(labels.shape[0], size=1000)
    #     target_class = np.random.randint(100, size=1000)
    #
    #     for i in range(1000):
    #         while target_class[i] == labels[attack_idx[i]]:
    #             target_class[i] = np.random.randint(100)
    #             print('change of target class')
    #
    #     with open('cifar100_talbf.txt', 'w') as f:
    #         for i in range(1000):
    #             f.write(str(target_class[i]) + ' ' + str(attack_idx[i]) + '\n')
    #
    #     print('attack info saved')
    #     attack_info = np.loadtxt(args.attack_info).astype(int)



    asr, pa_acc, n_bit, n_stop, param_lam, param_k_bits = [], [], [], [], [], []
    for i, (target_class, attack_idx) in enumerate(attack_info):

        # weights[i][0] = get_all_weights(model)

        start_time2 = time.time()
        print('Target class: ', target_class)
        print('Attack idx: ', attack_idx)
        source_class = int(labels[attack_idx])
        aux_idx = np.random.choice([i for i in range(len(labels)) if i != attack_idx], args.n_aux, replace=False)

        suc = False
        cur_k = args.init_k
        for search_k in range(args.max_search_k):
            cur_lam = args.init_lam
            for search_lam in range(args.max_search_lam):
                # print('k: ', str(cur_k), 'lambda: ', str(cur_lam))
                res = attack(auglag, mid_out, labels, labels_cuda, attack_idx,
                             target_class, source_class, aux_idx, cur_lam, cur_k, args)

                if res == -1:
                    print("Error[{0}]: Lambda:{1} K_bits:{2}".format(i, cur_lam, cur_k))
                    cur_lam = cur_lam / 2.0
                    continue
                elif res["suc"]:
                    auglag_new = res["auglag_new"]
                    n_stop.append(int(res["stop"]))
                    asr.append(int(res["suc"]))
                    pa_acc.append(res["pa_acc"])
                    n_bit.append(res["n_bit"])
                    param_lam.append(cur_lam)
                    param_k_bits.append(cur_k)
                    suc = True
                    break
                else:
                    auglag_new = res["auglag_new"]
                cur_lam = cur_lam / 2.0
            if suc:
                break
            cur_k = cur_k * 2.0

        if not suc:
            asr.append(0)
            n_stop.append(0)
            print("[{0}] Fail!".format(i))
        else:
            print("[{0}] PA-ACC:{1:.4f} Success:{2} N_flip:{3} Stop:{4} Lambda:{5} K:{6}".format(
                i, pa_acc[-1]*100, bool(asr[-1]), n_bit[-1], bool(n_stop[-1]), param_lam[-1], param_k_bits[-1]))

            # if (i+1) % 10 == 0:
            #     print("END[0] PA-ACC:{1:.4f} ASR:{2} N_flip:{3:.4f}".format(
            #         i, np.mean(pa_acc)*100, np.mean(asr)*100, np.mean(n_bit)))

            # print(torch.norm(auglag_new.w_twos.data.view(-1) - auglag.w_twos.data.view(-1), p=0).item())
            # print(weights[i][0])
            # print(auglag_new.w_twos.data.cpu().numpy())
            # print('-------------------')
            # print(auglag.w_twos.data.cpu().numpy())

        time_1 = time.time()

        auglag_arr_ori = auglag.w_twos.data.cpu().numpy()
        auglag_arr_new = auglag_new.w_twos.data.cpu().numpy()

        shape_0 = auglag.w_twos.data.cpu().numpy().shape[0]
        shape_1 = auglag.w_twos.data.cpu().numpy().shape[1]

        list_ori = []
        list_new = []

        for y in range(shape_0):
            for z in range(shape_1):
                binary_value_ori = np.array(auglag_arr_ori[y][z], dtype=np.int)
                binary_value_new = np.array(auglag_arr_new[y][z], dtype=np.int)

                # Convert the binary value to a string
                binary_string_ori = ''.join(map(str, binary_value_ori))
                binary_string_new = ''.join(map(str, binary_value_new))

                # Determine the sign and convert the binary value to an integer
                decimal_integer_ori = twos_comp(int(binary_string_ori, 2), len(binary_string_ori))
                decimal_integer_new = twos_comp(int(binary_string_new, 2), len(binary_string_new))

                list_ori.append(decimal_integer_ori)
                list_new.append(decimal_integer_new)

                if decimal_integer_ori != decimal_integer_new:
                    print(decimal_integer_ori, decimal_integer_new)
                    print(auglag.w_twos.data.cpu().numpy()[y][z], auglag_new.w_twos.data.cpu().numpy()[y][z])
                    print("-------------------")

        weights_all[0][0] = np.array(list_ori)
        weights_all[0][1] = np.array(list_new)

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

        print("Time_1: ", time.time() - time_1)

        # print(weights[i][0])
        # print(weights[i][1])

        # weights[i][1] = get_all_weights(model_new)


        print("Attack Time: ", time.time() - start_time2)
        print("Attack Time so far: ", time.time() - start_time)
        print("-----------------------------------")


    print("END Original_ACC:{0:.4f} PA_ACC:{1:.4f} ASR:{2:.2f} N_flip:{3:.4f}".format(
            acc_ori*100, np.mean(pa_acc)*100, np.mean(asr)*100, np.mean(n_bit)))

    end_time = time.time()
    print("Attack Time: ", end_time - start_time)



    directory = save_location

    # Check if the directory exists
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        os.makedirs(directory)

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


if __name__ == '__main__':
    main()
