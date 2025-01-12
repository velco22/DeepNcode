python -u attack_bfa.py --data_path "data/CIFAR100/" --dataset "cifar100" --arch "resnet18_cifar100" --bits 8 --outdir "results/cifar100/resnet18_quan8/" --bfa --n 100 --n_iter 2000 2>&1 | tee output_cifar100/bfa_res18_int8_output.txt
python -u attack_bfa.py --data_path "data/CIFAR100/" --dataset "cifar100" --arch "vgg11_bn_quan_cifar100" --bits 8 --outdir "results/cifar100/vgg11_quan8/" --bfa --n 100 --n_iter 4000 2>&1 | tee output_cifar100/bfa_vgg11_int8_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "resnet18_cifar100" --type 1 --bits 4 --outdir "results/cifar100/resnet18_quan4/" --n 100 --fc 18 2>&1 | tee output_cifar100/tbfa_t1_res18_int4_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "vgg11_bn_quan_cifar100" --type 1 --bits 4 --outdir "results/cifar100/vgg11_quan4/" --n 100 --fc 9 2>&1 | tee output_cifar100/tbfa_t1_vgg11_int4_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "resnet18_cifar100" --type 2 --bits 4 --outdir "results/cifar100/resnet18_quan4/" --n 100 --fc 18 2>&1 | tee output_cifar100/tbfa_t2_res18_int4_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "vgg11_bn_quan_cifar100" --type 2 --bits 4 --outdir "results/cifar100/vgg11_quan4/" --n 100 --fc 9 2>&1 | tee output_cifar100/tbfa_t2_vgg11_int4_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "resnet18_cifar100" --type 3 --bits 4 --outdir "results/cifar100/resnet18_quan4/" --n 100 --fc 18  2>&1 | tee output_cifar100/tbfa_t3_res18_int4_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "vgg11_bn_quan_cifar100" --type 3 --bits 4 --outdir "results/cifar100/vgg11_quan4/" --n 100 --fc 9 2>&1 | tee output_cifar100/tbfa_t3_vgg11_int4_output.txt
python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "resnet18_cifar100" --bits 4 --outdir "results/cifar100/resnet18_quan4/" --attack_info "cifar100_100_talbf.txt" --tc "0" --n 100 2>&1 | tee output_cifar100/talbf_res18_int4_output.txt
python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "vgg11_bn_quan_cifar100" --bits 8 --outdir "results/cifar100/vgg11_quan8/" --attack_info "cifar100_100_talbf.txt" --tc "0" --n 100 2>&1 | tee output_cifar100/talbf_vgg11_int8_output.txt
python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "vgg11_bn_quan_cifar100" --bits 4 --outdir "results/cifar100/vgg11_quan4/" --attack_info "cifar100_100_talbf.txt" --tc "0" --n 100 2>&1 | tee output_cifar100/talbf_vgg11_int4_output.txt
python -u attack_bfa.py --data_path "data/CIFAR100/" --dataset "cifar100" --arch "resnet18_cifar100" --bits 4 --outdir "results/cifar100/resnet18_quan4/" --bfa --n 100 --n_iter 2000 2>&1 | tee output_cifar100/bfa_res18_int4_output.txt
python -u attack_bfa.py --data_path "data/CIFAR100/" --dataset "cifar100" --arch "vgg11_bn_quan_cifar100" --bits 4 --outdir "results/cifar100/vgg11_quan4/" --bfa --n 100 --n_iter 4000 2>&1 | tee output_cifar100/bfa_vgg11_int4_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_4_quan" --type 1 --bits 4 --outdir "results/cifar100/wrn28_4_quan4/" --n 100 --fc 28 2>&1 | tee output_cifar100/tbfa_t1_wrn28_4_int4_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_4_quan" --type 1 --bits 8 --outdir "results/cifar100/wrn28_4_quan8/" --n 100 --fc 28 2>&1 | tee output_cifar100/tbfa_t1_wrn28_4_int8_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_4_quan" --type 2 --bits 4 --outdir "results/cifar100/wrn28_4_quan4/" --n 100 --fc 28 2>&1 | tee output_cifar100/tbfa_t2_wrn28_4_int4_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_4_quan" --type 2 --bits 8 --outdir "results/cifar100/wrn28_4_quan8/" --n 100 --fc 28 2>&1 | tee output_cifar100/tbfa_t2_wrn28_4_int8_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_4_quan" --type 3 --bits 4 --outdir "results/cifar100/wrn28_4_quan4/" --n 100 --fc 28 2>&1 | tee output_cifar100/tbfa_t3_wrn28_4_int4_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_4_quan" --type 3 --bits 8 --outdir "results/cifar100/wrn28_4_quan8/" --n 100 --fc 28 2>&1 | tee output_cifar100/tbfa_t3_wrn28_4_int8_output.txt
python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_4_quan" --bits 4 --outdir "results/cifar100/wrn28_4_quan4/" --attack_info "cifar100_100_talbf.txt" --tc "0" --n 100 2>&1 | tee output_cifar100/talbf_wrn28_4_int4_output.txt
python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_4_quan" --bits 8 --outdir "results/cifar100/wrn28_4_quan8/" --attack_info "cifar100_100_talbf.txt" --tc "0" --n 100 2>&1 | tee output_cifar100/talbf_wrn28_4_int8_output.txt
python -u attack_bfa.py --data_path "data/CIFAR100/" --dataset "cifar100" --arch "wrn28_4_quan" --bits 4 --outdir "results/cifar100/wrn28_4_quan4/" --bfa --n 100 --n_iter 2000 2>&1 | tee output_cifar100/bfa_wrn28_4_int4_output.txt
python -u attack_bfa.py --data_path "data/CIFAR100/" --dataset "cifar100" --arch "wrn28_4_quan" --bits 8 --outdir "results/cifar100/wrn28_4_quan8/" --bfa --n 100 --n_iter 4000 2>&1 | tee output_cifar100/bfa_wrn28_4_int8_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_8_quan" --type 1 --bits 4 --outdir "results/cifar100/wrn28_8_quan4/" --n 100 --fc 28 2>&1 | tee output_cifar100/tbfa_t1_wrn28_8_int4_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_8_quan" --type 1 --bits 8 --outdir "results/cifar100/wrn28_8_quan8/" --n 100 --fc 28 2>&1 | tee output_cifar100/tbfa_t1_wrn28_8_int8_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_8_quan" --type 2 --bits 4 --outdir "results/cifar100/wrn28_8_quan4/" --n 100 --fc 28 2>&1 | tee output_cifar100/tbfa_t2_wrn28_8_int4_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_8_quan" --type 2 --bits 8 --outdir "results/cifar100/wrn28_8_quan8/" --n 100 --fc 28 2>&1 | tee output_cifar100/tbfa_t2_wrn28_8_int8_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_8_quan" --type 3 --bits 4 --outdir "results/cifar100/wrn28_8_quan4/" --n 100 --fc 28 2>&1 | tee output_cifar100/tbfa_t3_wrn28_8_int4_output.txt
python -u attack_tbfa_changed.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_8_quan" --type 3 --bits 8 --outdir "results/cifar100/wrn28_8_quan8/" --n 100 --fc 28 2>&1 | tee output_cifar100/tbfa_t3_wrn28_8_int8_output.txt
python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_8_quan" --bits 4 --outdir "results/cifar100/wrn28_8_quan4/" --attack_info "cifar100_100_talbf.txt" --tc "0" --n 100 2>&1 | tee output_cifar100/talbf_wrn28_8_int4_output.txt
python -u attack_talbf.py --data_dir "data/" --dataset "CIFAR100" -c 100 --arch "wrn28_8_quan" --bits 8 --outdir "results/cifar100/wrn28_8_quan8/" --attack_info "cifar100_100_talbf.txt" --tc "0" --n 100 2>&1 | tee output_cifar100/talbf_wrn28_8_int8_output.txt
python -u attack_bfa.py --data_path "data/CIFAR100/" --dataset "cifar100" --arch "wrn28_8_quan" --bits 4 --outdir "results/cifar100/wrn28_8_quan4/" --bfa --n 100 --n_iter 2000 2>&1 | tee output_cifar100/bfa_wrn28_8_int4_output.txt
python -u attack_bfa.py --data_path "data/CIFAR100/" --dataset "cifar100" --arch "wrn28_8_quan" --bits 8 --outdir "results/cifar100/wrn28_8_quan8/" --bfa --n 100 --n_iter 4000 2>&1 | tee output_cifar100/bfa_wrn28_8_int8_output.txt