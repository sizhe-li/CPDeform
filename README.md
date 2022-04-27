# CPDeform
Code and data for paper [Contact Points Discovery for Soft-Body Manipulations with Differentiable Physics](https://lester0866.github.io/publication/contact_points_discovery_iclr2022/) at ICLR 2022 (Spotlight).

![Alt Text](https://github.com/lester0866/CPDeform/blob/main/demo/writer_demo.gif)

```bibtex
@InProceedings{li2022contact,
author = {Li, Sizhe and Huang, Zhiao and Du, Tao and Su, Hao and Tenenbaum, Joshua and Gan, Chuang},
title = {{C}ontact {P}oints {D}iscovery for {S}oft-{B}ody {M}anipulations with {D}ifferentiable {P}hysics},
booktitle = {International Conference on Learning Representations (ICLR)},
year = {2022}}
```

# Installation

```bash
python3 -m pip install -e .
conda install pyg -c pyg
```

# Data

Download target templates [here](https://drive.google.com/drive/folders/1Ym7XA-1_W1XZ9c0n8jJq04bpVbH2qTF8?usp=sharing),
and put the folder `diff_phys` that contains goal shapes onto your machine.

# Experiments

Run the following to train an agent that writes "ICLR" on a plasticine board:

```bash
python3 scripts/launch_training.py \
--root_dir somewhere_on_your_machine/diff_phys \
--algo cpdeform \ 
--env_name multistage_writer
```

The experiment results will be stored in the `diff_phys` folder. 

# Acknowledgements

Our physics simulation is based on [PlasticineLab](https://github.com/hzaskywalker/PlasticineLab).
