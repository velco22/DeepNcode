# DeepNcode: Encoding-Based Protection against Bit-Flip Attacks on Neural Networks
### Authors: Patrik Velčický, Jakub Breier, Mladen Kovačević, Xiaolu Hou
### Paper: https://arxiv.org/abs/2405.13891
### Master's Thesis Evidence No.: FIIT-183739-103175

To cite this work, please use the following BibTeX entry:
@article{velvcicky2024deepncode,
  title={DeepNcode: Encoding-Based Protection against Bit-Flip Attacks on Neural Networks},
  author={Vel{\v{c}}ick{\`y}, Patrik and Breier, Jakub and Kova{\v{c}}evi{\'c}, Mladen and Hou, Xiaolu},
  journal={arXiv preprint arXiv:2405.13891},
  year={2024}
}

## Important files:
*.sh files - examples of usage\
attack_bfa.py - code for BFA attack with simulation of our countermeasure\
attack_talbf.py - code for TA-LBF attack with simulation of our countermeasure\
attack_tbfa.py - code for T-BFA attack with simulation of our countermeasure\
main.py - used for training NN\
our_implementation_of_qnn - implementation of qnn with countermeasure

## Parts of this code repository is based on the following works:
BFA - https://github.com/elliothe/BFA \
TA-LBF - https://github.com/jiawangbai/TA-LBF, https://github.com/IGITUGraz/OutputCodeMatching \
T-BFA - https://github.com/adnansirajrakin/T-BFA, https://github.com/IGITUGraz/OutputCodeMatching
