python -u main.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "resnet18_cifar100" --bits 8 --outdir "results/cifar100/resnet18_quan8/" 2>&1 | tee output_cifar100_train/resnet18_quan8.txt
python -u main.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "vgg11_bn_quan_cifar100" --bits 4 --outdir "results/cifar100/vgg11_quan4/" 2>&1 | tee output_cifar100_train/vgg11_quan4.txt
python -u main.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "vgg11_bn_quan_cifar100" --bits 8 --outdir "results/cifar100/vgg11_quan8/" 2>&1 | tee output_cifar100_train/vgg11_quan8.txt
python -u main.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "resnet34_cifar100" --bits 4 --outdir "results/cifar100/resnet34_quan4/" 2>&1 | tee output_cifar100_train/resnet34_quan4.txt
python -u main.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "resnet34_cifar100" --bits 8 --outdir "results/cifar100/resnet34_quan8/" 2>&1 | tee output_cifar100_train/resnet34_quan8.txt
python -u main.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "vgg13_bn_quan_cifar100" --bits 4 --outdir "results/cifar100/vgg13_quan4/" 2>&1 | tee output_cifar100_train/vgg13_quan4.txt
python -u main.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "vgg13_bn_quan_cifar100" --bits 8 --outdir "results/cifar100/vgg13_quan8/" 2>&1 | tee output_cifar100_train/vgg13_quan8.txt


python -u main.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_8_quan" --bits 8 --outdir "results/cifar100/wrn28_8_quan8/"
python -u main.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_8_quan" --bits 4 --outdir "results/cifar100/wrn28_8_quan4/"
python -u main.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_4_quan" --bits 8 --outdir "results/cifar100/wrn28_4_quan8/"
python -u main.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_4_quan" --bits 4 --outdir "results/cifar100/wrn28_4_quan4/"
