python -u attack_bfa.py --data_path "data/CIFAR10/" --dataset "cifar10" --arch "resnet20_quan" --bits 4 --outdir "results/cifar10/resnet20_quan4/" --bfa
python -u attack_bfa.py --data_path "data/CIFAR10/" --dataset "cifar10" --arch "resnet20_quan" --bits 8 --outdir "results/cifar10/resnet20_quan8/" --bfa


python -u attack_bfa.py --data_path "data/CIFAR10/" --dataset "cifar10" --arch "vgg11_bn_quan_cifar10" --bits 4 --outdir "results/cifar10/vgg11_bn_quan4/" --bfa --n 2
python -u attack_bfa.py --data_path "data/CIFAR10/" --dataset "cifar10" --arch "vgg11_bn_quan_cifar10" --bits 8 --outdir "results/cifar10/vgg11_bn_quan8/" --bfa --n 2

python -u main.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --bits 8 --outdir "results/cifar10/vgg11_bn_quan8/"

python -u main.py --data_dir "data/" --dataset "ImageNet" -c 1000 --arch "resnet50_quan" --bits 8 -b 64 --outdir "results/imagenet/resnet50_quan8/"
python -u main.py --data_dir "data/" --dataset "ImageNet" -c 1000 --arch "resnet50_quan" --bits 4 -b 64 --outdir "results/imagenet/resnet50_quan4/"

python -u attack_talbf.py --data_dir "data/" --dataset "ImageNet" -c 1000 --arch "resnet18_quan" --bits 4 --outdir "results/imagenet/resnet18_quan4/" --attack_info "cifar10_talbf.txt" --tc "0" --n 2

# test vgg11
python -u attack_bfa.py --data_path "data/CIFAR10/" --dataset "cifar10" --arch "vgg11_bn_quan_cifar10" --bits 4 --outdir "results/cifar10/vgg11_bn_quan4/" --bfa --n 2
python -u attack_bfa.py --data_path "data/CIFAR10/" --dataset "cifar10" --arch "vgg11_bn_quan_cifar10" --bits 8 --outdir "results/cifar10/vgg11_bn_quan8/" --bfa --n 2

python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --bits 4 --outdir "results/cifar10/vgg11_bn_quan4/" --attack_info "cifar10_talbf.txt" --tc "0" --n 2
python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --bits 8 --outdir "results/cifar10/vgg11_bn_quan8/" --attack_info "cifar10_talbf.txt" --tc "0" --n 2

python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --type 1 --bits 4 --outdir "results/cifar10/vgg11_bn_quan4/" --n 2 --fc 9
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --type 1 --bits 8 --outdir "results/cifar10/vgg11_bn_quan8/" --n 2 --fc 9

python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --type 2 --bits 4 --outdir "results/cifar10/vgg11_bn_quan4/" --n 2 --fc 9
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --type 2 --bits 8 --outdir "results/cifar10/vgg11_bn_quan8/" --n 2 --fc 9

python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --type 3 --bits 4 --outdir "results/cifar10/vgg11_bn_quan4/" --n 2 --fc 9
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR10" -c 10 --arch "vgg11_bn_quan_cifar10" --type 3 --bits 8 --outdir "results/cifar10/vgg11_bn_quan8/" --n 2 --fc 9
