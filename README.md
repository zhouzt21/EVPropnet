# EVPropNet
EVPropNet: Detecting Drones By Finding Propellers For Mid-Air Landing And Following

**EVPropNet** by <a href="http://prg.cs.umd.edu"><i>Perception & Robotics Group</i></a> at the Department of Computer Science, <a href="https://umd.edu/">University of Maryland- College Park</a> and <a href="https://mavlab.tudelft.nl/"><i>Micro Air Vehicle Laboratory</i></a> at <a href="https://www.tudelft.nl/en/">Delft University of Technology</a>
.

## Code (Now Released)

### *For running the code, please refer to our [wiki](https://github.com/prgumd/EVPropNet/wiki).* 

Here's an example of EVPropNet output for drone detection and following:

![Detection And Following](https://github.com/prgumd/EVPropNet/blob/main/assests/Follow.gif)


***
![EVPropNet: Detecting Drones By Finding Propellers For Mid-Air Landing And Following](http://prg.cs.umd.edu/research/EVPropNet_files/Banner.jpg)
Applications presented in this work using the proposed propeller detection method for finding multi-rotors. (a) Tracking and following an unmarked quadrotor, (b) Landing/Docking on a flying quadrotor. Red and green arrows indicates the movement of the larger and smaller quadrotors respectively. Time progression is shown as quadrotor opacity. The insets show the event frames from the smaller quadrotor used for detecting the propellers of the bigger quadrotor using the proposed EVPropNet. Red and blue color in the event frames indicate positive and negative events respectively. Green color indicates the network prediction. Click on the image to zoom in.

## Abstract

The rapid rise of accessibility of unmanned aerial vehicles or drones pose a threat to general security and confidentiality. Most of the commercially available or custom-built drones are multi-rotors and are comprised of multiple propellers. Since these propellers rotate at a high-speed, they are generally the fastest moving parts of an image and cannot be directly "seen" by a classical camera without severe motion blur. We utilize a class of sensors that are particularly suitable for such scenarios called event cameras, which have a high temporal resolution, low-latency, and high dynamic range.

In this paper, we model the geometry of a propeller and use it to generate simulated events which are used to train a deep neural network called EVPropNet to detect propellers from the data of an event camera. EVPropNet directly transfers to the real world without any fine-tuning or retraining. We present two applications of our network: (a) tracking and following an unmarked drone and (b) landing on a near-hover drone. We successfully evaluate and demonstrate the proposed approach in many real-world experiments with different propeller shapes and sizes. Our network can detect propellers at a rate of 85.1% even when 60% of the propeller is occluded and can run at upto 35Hz on a 2W power budget. To our knowledge, this is the first deep learning-based solution for detecting propellers (to detect drones). Finally, our applications also show an impressive success rate of 92% and 90% for the tracking and landing tasks respectively.

- [Project Page](https://prg.cs.umd.edu/EVPropNet)
- [Paper](https://prg.cs.umd.edu/research/EVPropNet_files/EVPropNet.pdf)
- [arXiv Preprint](https://arxiv.org/abs/)


## License:
Copyright (c) 2021 Perception and Robotics Group (PRG).

## 运行

重新训练：
1. 下载背景图片
``` bash
apt update
sudo apt install aria2
cd /hy-tmp/
aria2c -x 16 -s 16 http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
```

2. 生成仿真无人机数据(1w张)
``` bash
git clone https://github.com/prgumd/EVPropNet.git
pip install numpy matplotlib tqdm termcolor imutils scikit-image
pip install opencv-python==3.4.18.65
python GenData.py --BgImgPath /hy-tmp/train2014/ --NumImages 10000 --WritePath /hy-tmp/DVSProp/
```

3. 训练Train Code
   
``` bash 
python Train.py --BasePath=/hy-tmp/DVSProp/ --NumEpochs=50 --MiniBatchSize=32 --CheckPointPath=/home/EVPropNet/DVSProp/CheckPoints/ --LogsPath=/home/EVPropNet/DVSProp/Logs/ --LR=1e-4 --GPUDevice 0 --ImgFormat png
```
断点续训：

``` bash
python Train.py --BasePath=/hy-tmp/DVSProp/ --NumEpochs=50 --MiniBatchSize=32 --CheckPointPath=/home/EVPropNet/DVSProp/CheckPoints/ --LogsPath=/home/EVPropNet/DVSProp/Logs/ --LR=1e-4 --GPUDevice 0 --ImgFormat png --LoadCheckPoint 1
```

4. 使用自己的测试集测试
For Simulated Data :
``` bash
python Test.py --CheckPointPath=/home/EVPropNet/DVSProp/CheckPoints/49model.ckpt --GPUDevice 0 --WritePath=/home/EVPropNet/DVSProp/Outputs/simulated_data/ --TestMode=M --Input=/hy-tmp/DVSPropTest/DVSPropTest/Imgs/ --ImgFormat=png
```

For Real Data:
``` bash
python Test.py --CheckPointPath=/home/EVPropNet/DVSProp/CheckPoints/49model.ckpt --GPUDevice 0 --WritePath=/home/EVPropNet/DVSProp/Outputs/real_data/ --TestMode=M --Input=/hy-tmp/RealData/RealData/ --ImgFormat=png --ImgSize [640,480,1]
```

For our data:
``` bash
python Test.py --CheckPointPath=/home/EVPropNet/DVSProp/CheckPoints/49model.ckpt --GPUDevice 0 --WritePath=/home/EVPropNet/DVSProp/Outputs/our_data/output_1113_100m/ --TestMode=M --Input=/hy-tmp/100000/output_1113_100m/ --ImgFormat=png --ImgSize [720,1280,1]
```

批量运行our data:
``` bash
 bash run_our_data.bash
```

1. 输出检测框
修改测试代码为：Test1.py

For Real Data:
``` bash 
python Test1.py --CheckPointPath=/home/EVPropNet/DVSProp/CheckPoints/49model.ckpt --GPUDevice 0 --WritePath=/home/EVPropNet/DVSProp/Outputs/real_data_rectangle/ --TestMode=M --Input=/hy-tmp/RealData/RealData/ --ImgFormat=png --ImgSize [640,480,1]
```
批量运行our data:
``` bash
 bash run_our_data_rectangle.bash
```
