import argparse
import random
import time
import datasets
import models
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, L1Loss
from torchvision import transforms
import torchvision
from xlwt import Workbook

from scipy.spatial.distance import hamming

## This script is adapted from the following public repository:
## https://github.com/adnansirajrakin/T-BFA

parser = argparse.ArgumentParser(description='Stealthy Targeted-BFA on DNNs')
parser.add_argument('--type', type=int, default=4, help='type of the attack')
parser.add_argument('--data_dir', type=str, default='data/')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset for processing')
parser.add_argument('--num_classes', '-c', default=10, type=int, help='number of classes in the dataset')
parser.add_argument('--arch', '-a', type=str, default='resnet20_quan', help='model architecture')
parser.add_argument('--bits', type=int, default=8, help='quantization bits')
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
parser.add_argument('--n', default=100, type=int, help='number_of_rounds')
parser.add_argument('--fc', type=int, default=20, help='number of layers before FC layer')

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


def data_gen():
    if args.dataset == 'CIFAR10':
        tr_test = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])
        testset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'CIFAR10'), train=False, transform=tr_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
        num_source, num_others = 1000, 9000
        im_size = 32
    elif args.dataset == 'CIFAR100':
        tr_test = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.267, 0.256, 0.276])])
        testset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'CIFAR100'), train=False, transform=tr_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
        num_source, num_others = 100, 9900
        im_size = 32
    elif args.dataset == 'ImageNet':
        tr_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        testset = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'imagenet/validation'), tr_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        num_source, num_others = 50, 49950
        im_size = 224
    else:
        print('Dataset not implemented, will crash soon...')
        pass

    # dataT and targetT will contain only one class images whose image will be missclassified
    data, dataT = torch.zeros([num_others, 3, im_size, im_size]).cuda(), torch.zeros([num_source, 3, im_size, im_size]).cuda()
    target, targetT = torch.zeros([num_others]).long().cuda(), torch.zeros([num_source]).long().cuda()
    xs, xn = 0, 0
    for t, (x, y) in enumerate(test_loader):
        if t < num_others + num_source:
            if y != args.source:
                data[xs, :, :, :] = x[0, :, :, :]
                target[xs] = y.long()
                xs += 1
            if y == args.source:
                dataT[xn, :, :, :] = x[0, :, :, :]
                targetT[xn] = y.long()
                xn += 1

    data1, target1 = data[0:args.auxiliary, :, :, :], target[0:args.auxiliary]          # only separating validation samples
    data2, target2 = data[args.auxiliary:, :, :, :], target[args.auxiliary:]            # separating the rest
    dataT1, targetT1 = dataT[0:args.attacksamp, :, :, :], targetT[0:args.attacksamp]    # separating "to be attacked" test samples
    dataT2, targetT2 = dataT[num_source - args.attacksamp:num_source, :, :, :], targetT[num_source - args.attacksamp:num_source]  # separating the rest from source class

    ##  random test batch for type 1 attack
    if args.dataset == 'CIFAR10':
        batch_size = 1000
    elif args.dataset == 'CIFAR100':
        batch_size = 1000
    elif args.dataset == 'ImageNet':
        batch_size = 50

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    for t, (x, y) in enumerate(test_loader):
        xx, yy = x.cuda(), y.cuda()
        break
    yy[:] = args.target



    return data1, target1, data2, target2, dataT1, targetT1, dataT2, targetT2, xx, yy



def validate(model, loader, C):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            if args.ocm:
                probs = F.softmax(torch.log(F.relu(torch.matmul(model(data), C.T)) + 1e-6))
                labels = torch.tensor([torch.where(torch.all(C == target[i], dim=1))[0][0] for i in range(target.shape[0])])
                pred = probs.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(labels.to(device).view_as(pred)).sum().item()
            else:
                output = nn.Softmax()(model(data))
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
    # print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, loader.sampler.__len__(), 100. * correct / loader.sampler.__len__()))
    return 100. * correct / loader.sampler.__len__()


def validate2(model, loader, to_attack):
    "this function computes the attack success rate of  all data to target class toattack"
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            target[:] = to_attack

            output = nn.Softmax()(model(data))
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, loader.sampler.__len__(), 100. * correct / loader.sampler.__len__()))
    return 100. * correct / loader.sampler.__len__()

def validate_batchwise(model, input, label, C):
    # validate3
    """ this function computes the accuracy of a given input and label batchwise """
    model.eval()
    correct, n, m = 0, 0, 100
    with torch.no_grad():
        for i in range((input.shape[0]) // 100):
            data, target = input[n:m, :, :, :].cuda(), label[n:m].cuda()
            m += 100
            n += 100
            if args.ocm:
                probs = F.softmax(torch.log(F.relu(torch.matmul(model(data), C.T)) + 1e-6))
                pred = probs.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                output = nn.Softmax()(model(data))
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
    # print('\nSub Test set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, input.shape[0], 100. * correct / input.shape[0]))
    return 100. * correct / input.shape[0]


def validate_batchwise_asr(model, data, target, xn, C):
    # validate1
    """ this function computes the accuracy for a given data and target on model """
    model.eval()
    correct = 0
    with torch.no_grad():
        data, target = data.cuda(), target.cuda()
        if args.ocm:
            probs = F.softmax(torch.log(F.relu(torch.matmul(model(data), C.T)) + 1e-6))
            pred = probs.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        else:
            output = nn.Softmax()(model(data))
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print('\nSubTest set: Attack Success Rate: {}/{} ({:.4f}%)\n'.format(correct, xn, 100. * correct / xn))
    return 100. * correct / xn


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
def main():
    # Load dataset
    DATASET = datasets.__dict__[args.dataset](args)
    train_loader, test_loader = DATASET.loaders()

    if args.dataset == 'CIFAR10':
        args.attacksamp, args.auxiliary = 500, 500
        # source_list = list(range(10))[args.source_start:args.source_end]
        # target_list = []
        # for s in source_list:
        #     if s == 0:
        #         target_list.append([9, 1])
        #     elif s == 9:
        #         target_list.append([8, 0])
        #     else:
        #         target_list.append([s - 1, s + 1])

        source_list = [random.randint(0, 9) for _ in range(args.n)]
        target_list = [[random.randint(0, 9)] for _ in range(args.n)]
        for i in range(args.n):
            while target_list[i][0] == source_list[i]:
                # print('source: ', source_list[i], 'target: ', target_list[i][0], 'index:', i)
                target_list[i][0] = random.randint(0, 9)
        #         print('source: ', source_list[i], 'target: ', target_list[i][0], 'index:', i)
        #         print('-------------------')
        # print(source_list)
        # print(target_list)
        # time.sleep(10)

    elif args.dataset == 'CIFAR100' or args.dataset == 'ImageNet':
        args.attacksamp = 50 if args.dataset == 'CIFAR100' else 25
        args.auxiliary = 50 if args.dataset == 'CIFAR100' else 25
        # source_list = list(range(50))[args.source_start:args.source_end]
        # target_list = []
        # for s in source_list:
        #     if s == 0:
        #         target_list.append([49, 1])
        #     elif s == 49:
        #         target_list.append([48, 0])
        #     else:
        #         target_list.append([s - 1, s + 1])
        if args.dataset == 'CIFAR100':
            source_list = [random.randint(0, 99) for _ in range(args.n)]
            target_list = [[random.randint(0, 99)] for _ in range(args.n)]
            for i in range(args.n):
                while target_list[i][0] == source_list[i]:
                    # print('source: ', source_list[i], 'target: ', target_list[i][0], 'index:', i)
                    target_list[i][0] = random.randint(0, 99)

        elif args.dataset == 'ImageNet':
            source_list = [random.randint(0, 999) for _ in range(args.n)]
            target_list = [[random.randint(0, 999)] for _ in range(args.n)]
            for i in range(args.n):
                while target_list[i][0] == source_list[i]:
                    # print('source: ', source_list[i], 'target: ', target_list[i][0], 'index:', i)
                    target_list[i][0] = random.randint(0, 999)

    # Load model architecture
    if args.ocm:
        n_output = args.code_length
        criterion = L1Loss()
        C = torch.tensor(DATASET.C).to(device)
    else:
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

    print(model)

    weight_conversion(model)
    benign_test_acc = validate(model, test_loader, C)
    print(benign_test_acc)

    start_time = time.time()

    # source_list = list([8])
    # target_list = list([[1]])
    args.avgs = 1

    weights = np.empty((args.n, 2), dtype=np.ndarray)
    weights_all = np.empty((1, 2), dtype=np.ndarray)

    acc = torch.Tensor(args.n, args.iters + 1).fill_(0)  # accuracy tracker
    acc1 = torch.Tensor(args.n, args.iters + 1).fill_(0)  # accuracy without attacked class and test samples used for attack
    temp = torch.Tensor(args.n, args.iters + 1).fill_(0)  # ASR on attack samples
    temp1 = torch.Tensor(args.n, args.iters + 1).fill_(0)  # ASR on rest of the samples
    layer = torch.Tensor(args.n, args.iters + 1).fill_(0)
    offsets = torch.Tensor(args.n, args.iters + 1).fill_(0)
    bfas = torch.Tensor(args.n).fill_(0)  # recording number of bit-flips

    round_now = 0
    save_location = './tbfa_results/' + args.outdir[8:-1] + "_" + str(args.n) + "_type" + str(args.type) + '/'

    directory = save_location

    # Check if the directory exists
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        os.makedirs(directory)

    for (source, targets) in zip(source_list, target_list):
        print('round: ', round_now)
        for target in targets:
            # print('source class: ', str(source), 'target class: ', str(target))
            args.source, args.target = source, target
            rounds = args.iters  # attack iterations

            ## type 1 attack
            if args.type == 1:


                for j in range(args.avgs):
                    # print("\n T-BFA Type1 Attack Repetition {}".format(j))


                    attacker = SneakyBFA(criterion, C, args.fc)
                    model.load_state_dict(state_dict)
                    weight_conversion(model)

                    weights_all[0][0] = get_all_weights(model)

                    model = model.cuda()
                    model.eval()

                    data1, target1, data2, target2, dataT1, targetT1, dataT2, targetT2, xx, yy = data_gen()

                    # print("data1: ", data1.shape)
                    # print("target1: ", target1.shape)
                    # print("target1: ", target1.long())
                    # print("xx: ", xx)
                    # print("xx: ", xx.shape)
                    # print("yy: ", yy)
                    # print("yy: ", yy.shape)
                    # time.sleep(10)

                    acc[round_now, 0] = validate(model, test_loader, C)
                    temp[round_now, 0] = validate2(model, test_loader, args.target)

                    for r in range(rounds):
                        layer[round_now, r + 1], offsets[round_now, r + 1] = attacker.progressive_bit_search12(model, xx, yy, data1, target1.long())
                        # print(r + 1)
                        acc[round_now, r + 1] = validate(model, test_loader, C)
                        temp[round_now, r + 1] = validate2(model, test_loader, args.target)

                        if temp[round_now, r + 1] > 99.99:
                            break
                    bfas[round_now] = int(r + 1)

                    weights_all[0][1] = get_all_weights(model)

                    dif = compare_weights(weights_all)

                    weights[round_now][0] = dif[0][0]
                    weights[round_now][1] = dif[0][1]

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

            ## type 2 attack
            if args.type == 2:

                # weights = np.empty((args.avgs, 2), dtype=np.ndarray)

                for j in range(args.avgs):
                    # print("\n T-BFA Type2 Attack Repetition {}".format(j))

                    attacker = SneakyBFA(criterion, C, args.fc)
                    model.load_state_dict(state_dict)
                    weight_conversion(model)

                    weights_all[0][0] = get_all_weights(model)

                    model = model.cuda()
                    model.eval()
                    data1, target1, data2, target2, dataT1, targetT1, dataT2, targetT2, xx, yy = data_gen()

                    targetT1[:], targetT2[:] = args.target, args.target
                    acc[round_now, 0] = validate(model, test_loader, C)
                    acc1[round_now, 0] = validate_batchwise(model, data2, target2, C)
                    temp[round_now, 0] = validate_batchwise_asr(model, dataT1, targetT1.long(), args.attacksamp, C)
                    temp1[round_now, 0] = validate_batchwise_asr(model, dataT2, targetT2.long(), args.attacksamp, C)

                    for r in range(rounds):
                        layer[round_now, r + 1], offsets[round_now, r + 1] = attacker.progressive_bit_search12(model, dataT1,targetT1.long(), data1, target1.long())
                        # print(r + 1)
                        acc[round_now, r + 1] = validate(model, test_loader, C)
                        acc1[round_now, r + 1] = validate_batchwise(model, data2, target2, C)
                        temp[round_now, r + 1] = validate_batchwise_asr(model, dataT1, targetT1.long(), args.attacksamp, C)
                        temp1[round_now, r + 1] = validate_batchwise_asr(model, dataT2, targetT2.long(), args.attacksamp, C)
                        if float(temp1[round_now, r + 1]) > 99.99:
                            break
                    bfas[round_now] = int(r + 1)

                    weights_all[0][1] = get_all_weights(model)

                    dif = compare_weights(weights_all)

                    weights[round_now][0] = dif[0][0]
                    weights[round_now][1] = dif[0][1]

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


            ## type 3 attack
            if args.type == 3:

                # weights = np.empty((args.avgs, 2), dtype=np.ndarray)

                for j in range(args.avgs):

                    # print("\n T-BFA Type3 Attack Repetition {}".format(j))

                    attacker = SneakyBFA(criterion, C, args.fc)
                    model.load_state_dict(state_dict)
                    weight_conversion(model)

                    weights_all[0][0] = get_all_weights(model)

                    model = model.cuda()

                    data1, target1, data2, target2, dataT1, targetT1, dataT2, targetT2, xx, yy = data_gen()
                    targetT1[:], targetT2[:] = args.target, args.target
                    acc[round_now, 0] = validate(model, test_loader, C)
                    acc1[round_now, 0] = validate_batchwise(model, data2, target2, C)
                    temp[round_now, 0] = validate_batchwise_asr(model, dataT1, targetT1.long(), args.attacksamp, C)
                    temp1[round_now, 0] = validate_batchwise_asr(model, dataT2, targetT2.long(), args.attacksamp, C)

                    for r in range(rounds):
                        layer[round_now, r + 1], offsets[round_now, r + 1] = attacker.progressive_bit_search3(model, dataT1, targetT1.long(),data1,target1.long())
                        acc[round_now, r + 1] = validate(model, test_loader, C)
                        acc1[round_now, r + 1] = validate_batchwise(model, data2, target2, C)
                        temp[round_now, r + 1] = validate_batchwise_asr(model, dataT1, targetT1.long(), args.attacksamp, C)
                        temp1[round_now, r + 1] = validate_batchwise_asr(model, dataT2, targetT2.long(), args.attacksamp, C)
                        if float(temp1[round_now, r + 1]) > 99.9:
                            break
                        if r > 1:
                            if temp1[round_now, r - 1] == temp1[round_now, r + 1]:
                                break
                    bfas[round_now] = int(r + 1)

                    weights_all[0][1] = get_all_weights(model)

                    dif = compare_weights(weights_all)

                    weights[round_now][0] = dif[0][0]
                    weights[round_now][1] = dif[0][1]

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

            ## type 4 attack (basic from talbf paper)
            if args.type == 4:

                # weights = np.empty((args.avgs, 2), dtype=np.ndarray)

                for j in range(args.avgs):
                    print("\nStealthy T-BFA Attack Repetition {}".format(j))

                    attacker = SneakyBFA(criterion, C, args.fc)
                    model.load_state_dict(state_dict)
                    weight_conversion(model)

                    weights_all[0][0] = get_all_weights(model)

                    model = model.cuda()

                    data1, target1, data2, target2, dataT1, targetT1, dataT2, targetT2, xx, yy = data_gen()
                    targetT1[:], targetT2[:] = args.target, args.target
                    acc[j, 0] = validate(model, test_loader, C)
                    acc1[j, 0] = validate_batchwise(model, data2, target2, C)
                    temp[j, 0] = validate_batchwise_asr(model, dataT1, targetT1.long(), args.attacksamp, C)
                    temp1[j, 0] = validate_batchwise_asr(model, dataT2, targetT2.long(), args.attacksamp, C)

                    for r in range(rounds):
                        layer[j, r + 1], offsets[j, r + 1] = attacker.progressive_bit_search4(model, dataT1, targetT1.long(), data1, target1.long(), args)
                        acc[j, r + 1] = validate(model, test_loader, C)
                        acc1[j, r + 1] = validate_batchwise(model, data2, target2, C)
                        temp[j, r + 1] = validate_batchwise_asr(model, dataT1, targetT1.long(), args.attacksamp, C)
                        temp1[j, r + 1] = validate_batchwise_asr(model, dataT2, targetT2.long(), args.attacksamp, C)
                        if float(temp1[j, r + 1]) > 99.9 or float(temp[j, r + 1]) > 99.9:
                            acc[j, r + 1] = validate(model, test_loader, C)
                            acc1[j, r + 1] = validate_batchwise(model, data2, target2, C)
                            break
                        if attacker.random_flip_flag:
                            acc[j, r + 1] = validate(model, test_loader, C)
                            acc1[j, r + 1] = validate_batchwise(model, data2, target2, C)
                            break
                        if r > 1 and offsets[j, r - 1] == offsets[j, r + 1] and offsets[j, r - 1] == offsets[j, r]:
                            if layer[j, r - 1] == layer[j, r + 1] and layer[j, r - 1] == layer[j, r]:
                                acc[j, r + 1] = validate(model, test_loader, C)
                                acc1[j, r + 1] = validate_batchwise(model, data2, target2, C)
                                acc[j, r] = acc[j, r + 1]
                                acc[j, r - 1] = acc[j, r + 1]
                                acc1[j, r] = acc1[j, r + 1]
                                acc1[j, r - 1] = acc1[j, r + 1]
                                break
                    bfas[j] = int(r + 1)

                    weights_all[0][1] = get_all_weights(model)

                    dif = compare_weights(weights_all)

                    weights[round_now][0] = dif[0][0]
                    weights[round_now][1] = dif[0][1]

                    del data1, target1, data2, target2, dataT1, targetT1, dataT2, targetT2

        round_now += 1

    test_acc = torch.Tensor(args.n).fill_(0)     # overall test accuracy
    ASR_as = torch.Tensor(args.n).fill_(0)       # ASR
    ASR_val = torch.Tensor(args.n).fill_(0)      # validation ASR (for generalization)
    rem_acc = torch.Tensor(args.n).fill_(0)      # accuracy on remaining data (source class and val set not present)

    for i in range(args.n):
        test_acc[i] = acc[i, int(bfas[i])]
        ASR_as[i] = temp[i, int(bfas[i])]
        ASR_val[i] = temp1[i, int(bfas[i])]
        rem_acc[i] = acc1[i, int(bfas[i])]

    print(test_acc)
    print(ASR_as)
    print(ASR_val)
    print(rem_acc)
    print(bfas.mean())
    print(bfas.std())

    end_time = time.time()

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



    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet1')
    sheet1.write(0, 0, ("Test_ACC"))
    sheet1.write(0, 1, ("ASR_AS "))
    sheet1.write(0, 2, ("ASR_rest "))
    sheet1.write(0, 3, ("layer number"))
    sheet1.write(0, 4, ("offset"))
    sheet1.write(0, 5, ("PA_ACC"))
    count = 0
    for j in range(args.n):
        for p in range(int(bfas[j]) + 1):
            sheet1.write(p + 1 + count, 0, float(acc[j, p]))
            sheet1.write(p + 1 + count, 1, float(temp[j, p]))
            sheet1.write(p + 1 + count, 2, float(temp1[j, p]))
            sheet1.write(p + 1 + count, 3, float(layer[j, p]))
            sheet1.write(p + 1 + count, 4, float(offsets[j, p]))
            sheet1.write(p + 1 + count, 5, float(acc1[j, p]))
        count += int(bfas[j]) + 2

    sheet2 = wb.add_sheet('Sheet2')
    sheet2.write(0, 0, ("Test_ACC"))
    sheet2.write(0, 1, ("PA_ACC"))
    sheet2.write(0, 2, ("ASR_AS "))
    sheet2.write(0, 3, ("ASR_rest "))
    sheet2.write(0, 4, ("Bitflips "))
    sheet2.write(1, 0, float(test_acc.mean()))
    sheet2.write(2, 0, float(test_acc.std()))
    sheet2.write(1, 1, float(rem_acc.mean()))
    sheet2.write(2, 1, float(rem_acc.std()))
    sheet2.write(1, 2, float(ASR_as.mean()))
    sheet2.write(2, 2, float(ASR_as.std()))
    sheet2.write(1, 3, float(ASR_val.mean()))
    sheet2.write(2, 3, float(ASR_val.std()))
    sheet2.write(1, 4, float(bfas.mean()))
    sheet2.write(2, 4, float(bfas.std()))
    # file_string = "tbfa_results/" + args.outdir[8:-1] + "_type_" + str(args.type)
    file_string = save_location + "tables"
    wb.save(file_string + ".xls")



    print('Attack time: ', end_time - start_time)

if __name__ == "__main__":
    main()
