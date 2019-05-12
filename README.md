## Description
This work is used for reproduce MTCNN,a Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks.


## Demo usage
（说明：使用python + tensorflow 实现比较方便，故此版本使用python写的，后期可改为C++）

1. 搭建tensorflow, 参考https://www.tensorflow.org/install
2. 安装python包: opencv, numpy
3. python ./facedetect_mtcnn.py --input ./test.jpg --output  new.jpg


## Results
![image](https://github.com/qwer10/MTCNN_tf/blob/master/new.jpg)


## 论文要点
1. 利用face detection和 face alignment 这两种任务之间的共性，通过共享特征，一个CNN同时干两件事。
2. 级联,作者设计的CNN分3个stage（能快速产生候选窗口的浅层CNN --feed--> 能剔除大量错误数据的稍复杂的CNN --feed--> 能标出landmark复杂CNN）。这样性能提高很多。
3. Online hard sample mining。反向传播只计算代价最高的70%的样本的梯度下降。提高了性能。
4. 总的学习损失根据任务重要性动态计算。（在不同stages 组合 face/non-face classification + bounding box regression + facial landmark localization这三者的损失 ）
5. 优化filters，使用更小更多样的过滤器。

## 打算如何实现？
0. 首先根据论文建立P R O 三个网络模型

1. 建立图像金字塔
将原图尺寸反复乘以缩放因子，以便得到更多的boundingbox。

2. 将得到不同尺寸的图片放入到PNet中做前向运算
将上一步的N个缩放后的图片，放到PNet网络，得到boundingboxex

3. 用RNet筛选候选的boundingbox
首先按照每个候选框坐标信息在原图中获取数据，并且将所有的数据imResample到（24，24）的尺寸，方便送入到RNet网络中做前向运算。筛选掉一部分候选框。

4. ONet对RNet的输出作进一步处理，得到唯一的候选框
首先按照每个候选框坐标信息在原图中获取数据，并且将所有的数据imResample到（48，48）的尺寸，方便送入到ONet网络中做前向运算。筛选掉一部分候选框,得到唯一的人脸框。
根据ONet的五个特征点坐标值的输出，按照代码对应关系，得到原图对应的五个关键点的坐标值。

## Train
正样本：IoU >= 0.65
负样本：IoU < 0.3
部分(part)样本：0.65 > IoU >= 0.4
landmark样本

正负样本用于face classification tasks
负样本和part样本用于bounding box regression
landmark样本 用于facial landmark localization
landmark faces are used for facial landmark localization. 

