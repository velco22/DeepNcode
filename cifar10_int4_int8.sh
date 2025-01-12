#!/bin/bash
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt

data="CIFAR10"
dir="data/"
model="resnet20_quan"
classes=10

# Train ResNet-20 models (8-bit) with output code matching on CIFAR-10
python -u main.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/"
python -u main.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/"

# Train VGG-11 with batch norm models (8-bit) with output code matching on CIFAR-10
python -u main.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --bits 4 --outdir "results/cifar10/vgg11_bn_quan4/"
python -u main.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --bits 8 --outdir "results/cifar10/vgg11_bn_quan8/"

# Evaluate Stealthy T-BFA attacks on OCM defended models
python -u attack_tbfa.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/"
python -u attack_tbfa.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/"

# Evaluate Stealthy TA-LBF attacks on OCM defended models
python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/" --attack_info "cifar10_talbf.txt" --tc "0"
python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/" --attack_info "cifar10_talbf.txt" --tc "0"

python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --bits 4 --outdir "results/cifar10/vgg11_bn_quan4/" --attack_info "cifar10_talbf.txt" --tc "0"
python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --bits 8 --outdir "results/cifar10/vgg11_bn_quan8/" --attack_info "cifar10_talbf.txt" --tc "0"


python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --type 1 --bits 4 --outdir "results/cifar10/resnet20_quan4/"
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --type 1 --bits 8 --outdir "results/cifar10/resnet20_quan8/"

python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --type 2 --bits 4 --outdir "results/cifar10/resnet20_quan4/"
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --type 2 --bits 8 --outdir "results/cifar10/resnet20_quan8/"

python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --type 3 --bits 4 --outdir "results/cifar10/resnet20_quan4/"
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --type 3 --bits 8 --outdir "results/cifar10/resnet20_quan8/"



# python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/" --attack_info "cifar10_talbf_0.txt" --tc "0"
# python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/" --attack_info "cifar10_talbf_1.txt" --tc "1"

#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/" --attack_info "cifar10_talbf_2.txt" --tc "2"
#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/" --attack_info "cifar10_talbf_3.txt" --tc "3"
#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/" --attack_info "cifar10_talbf_4.txt" --tc "4"

#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/" --attack_info "cifar10_talbf_5.txt" --tc "5"
#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/" --attack_info "cifar10_talbf_6.txt" --tc "6"

#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/" --attack_info "cifar10_talbf_7.txt" --tc "7"
#
#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/" --attack_info "cifar10_talbf_8.txt" --tc "8"
#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/" --attack_info "cifar10_talbf_9.txt" --tc "9"





#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/" --attack_info "cifar10_talbf_0.txt" --tc "0"

#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/" --attack_info "cifar10_talbf_1.txt" --tc "1"

#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/" --attack_info "cifar10_talbf_2.txt" --tc "2"

#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/" --attack_info "cifar10_talbf_3.txt" --tc "3"

#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/" --attack_info "cifar10_talbf_4.txt" --tc "4"
#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/" --attack_info "cifar10_talbf_5.txt" --tc "5"

#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/" --attack_info "cifar10_talbf_6.txt" --tc "6"
#python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/" --attack_info "cifar10_talbf_7.txt" --tc "7"

python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/" --attack_info "cifar10_talbf_8.txt" --tc "8"
python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/" --attack_info "cifar10_talbf_9.txt" --tc "9"
