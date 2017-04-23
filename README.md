# Reimplementation of CycleGAN model in pytorch:
### Authors code and paper:  [[torch]](https://github.com/junyanz/CycleGAN)[[pytorch]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)[[paper]](https://arxiv.org/pdf/1703.10593.pdf)

Learning image-to-image translation without input-output pairs.

### Horse -> Zebra 

### Separate updates of A and B generatos
![loss_separate](./pictures/separate_horse2zebra.png)
![example0](./pictures/imAB_gen_99_0_separate.jpg)
![example1](./pictures/imAB_gen_99_500_separate.jpg)
![example2](./pictures/imAB_gen_99_1000_separate.jpg)

### Fused updates of A and B generators 
![loss_fused](./pictures/fused_horse2zebra.png)
![example0](./pictures/imAB_gen_99_0_fused.jpg)
![example1](./pictures/imAB_gen_99_500_fused.jpg)
![example2](./pictures/imAB_gen_99_1000_fused.jpg)