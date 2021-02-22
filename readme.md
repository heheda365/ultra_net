# Ultra_net : A FPGA-based Object Detection for the DAC-SDC 2020

This is a repository for FPGA-based neural network inference. The design won first place in [the 57th IEEE/ACM Design Automation Conference System Design Contest (DAC-SDC)](http://https://dac-sdc-2020.groups.et.byu.net/doku.php).Designed by:
> BJUT_runner Group, Beijing University of Technology

> Kang ZHAN, Junnan GUO, Bingyan SONG, Wenbo ZHANG*, Zhenshan BAO*


![picture](pic/rank.jpg)
The final rankings are published at https://dac-sdc-2020.groups.et.byu.net/doku.php?id=results

![picture](pic/27.png)
![picture](pic/245.png)
![picture](pic/15.png)
![picture](pic/257.png)

## Repository Organization
- train: Contains the training script.
- model: Contains the pre-trained weights, model script and test script.
- quantization: Contains the python script which process the model generation header files to be used in Vivado HLS.
- hls: Contains the Vivado HLS implementation of Ultra_net.
- vivado: Contains the Vivado Block Design files.
- deploy: A Jupyter notebook showing how to use the FPGA based neural network to perform object detection on ultra96-PYNQ.

## train 
- cd train/yolov3/
- python3 train.py --multi-scale --img-size 320 --multi-scale --batch-size 32

## quantization
- cd quantization/
- python3 torch_export.py
- python3 ultranet_param_gen.py






