python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/" --n 1000
python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/" --n 1000
python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --bits 4 --outdir "results/cifar10/vgg11_bn_quan4/" --n 1000
python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --bits 8 --outdir "results/cifar10/vgg11_bn_quan8/" --n 1000
python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "resnet18_cifar100" --bits 4 --outdir "results/cifar100/resnet18_quan4/" --n 1000
python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "resnet18_cifar100" --bits 8 --outdir "results/cifar100/resnet18_quan8/" --n 1000
python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "vgg11_bn_quan_cifar100" --bits 4 --outdir "results/cifar100/vgg11_quan4/" --n 1000
python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "vgg11_bn_quan_cifar100" --bits 8 --outdir "results/cifar100/vgg11_quan8/" --n 1000
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_4_quan" --bits 4 --outdir "results/cifar100/wrn28_4_quan4/" --n 1000
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_4_quan" --bits 8 --outdir "results/cifar100/wrn28_4_quan8/" --n 1000
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_8_quan" --bits 4 --outdir "results/cifar100/wrn28_8_quan4/" --n 1000
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_8_quan" --bits 8 --outdir "results/cifar100/wrn28_8_quan8/" --n 1000
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/" --gpu "-1" --n 1
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/" --gpu "-1" --n 1
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --bits 4 --outdir "results/cifar10/vgg11_bn_quan4/" --gpu "-1" --n 1
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --bits 8 --outdir "results/cifar10/vgg11_bn_quan8/" --gpu "-1" --n 1
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "resnet18_cifar100" --bits 4 --outdir "results/cifar100/resnet18_quan4/" --gpu "-1" --n 1
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "resnet18_cifar100" --bits 8 --outdir "results/cifar100/resnet18_quan8/" --gpu "-1" --n 1
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "vgg11_bn_quan_cifar100" --bits 4 --outdir "results/cifar100/vgg11_quan4/" --gpu "-1" --n 1
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "vgg11_bn_quan_cifar100" --bits 8 --outdir "results/cifar100/vgg11_quan8/" --gpu "-1" --n 1
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_4_quan" --bits 4 --outdir "results/cifar100/wrn28_4_quan4/" --gpu "-1" --n 1
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_4_quan" --bits 8 --outdir "results/cifar100/wrn28_4_quan8/" --gpu "-1" --n 1
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_8_quan" --bits 4 --outdir "results/cifar100/wrn28_8_quan4/" --gpu "-1" --n 1
#python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_8_quan" --bits 8 --outdir "results/cifar100/wrn28_8_quan8/" --gpu "-1" --n 1


python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/" --n 0
python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/" --n 0
python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --bits 4 --outdir "results/cifar10/vgg11_bn_quan4/" --n 0
python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --bits 8 --outdir "results/cifar10/vgg11_bn_quan8/" --n 0
python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "resnet18_cifar100" --bits 4 --outdir "results/cifar100/resnet18_quan4/" --n 0
python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "resnet18_cifar100" --bits 8 --outdir "results/cifar100/resnet18_quan8/" --n 0
python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "vgg11_bn_quan_cifar100" --bits 4 --outdir "results/cifar100/vgg11_quan4/" --n 0
python -u overhead_calculation.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "vgg11_bn_quan_cifar100" --bits 8 --outdir "results/cifar100/vgg11_quan8/" --n 0
