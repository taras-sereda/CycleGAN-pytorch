# Reimplementation of CycleGAN model in pytorch:
[[paper]](https://arxiv.org/pdf/1703.10593.pdf) [[authors implementation(torch)]](https://github.com/junyanz/CycleGAN)

Learning image-to-image translation without input-output pairs.

### Horse -> Zebra 

###Separate updates of A and B generatos
![loss_separate](./pictures/separate_horse2zebra.png)
![example0](./pictures/imAB_gen_99_0_separate.jpg)
![example1](./pictures/imAB_gen_99_500_separate.jpg)
![example2](./pictures/imAB_gen_99_1000_separate.jpg)

###Fused updates of A and B generators 
![loss_fused](./pictures/fused_horse2zebra.png)
![example0](./pictures/imAB_gen_99_0_fused.jpg)
![example1](./pictures/imAB_gen_99_500_fused.jpg)
![example2](./pictures/imAB_gen_99_1000_fused.jpg)