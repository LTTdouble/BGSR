# BGSR
A Band Grouping-based Hybrid Convolution for Hyperspectral Image Super-Resolution

This repository is implementation of the ["A Band Grouping-based Hybrid Convolution for Hyperspectral Image Super-Resolution"](BGSR)by PyTorch.

Dataset
------
**Three public datasets, i.e., 
"CAVE"),[CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ 
"CAVE"), [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ 
"Harvard"), [Harvard](http://vision.seas.harvard.edu/hyperspec/download.html
"Chikusei"), [Chikusei](https://naotoyokoya.com/Download.html
"Houston"), [Houston](http://www.grss-ieee.org/community
"Pavia center"), [Pavia center](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_University_scene), 

are employed to verify the effectiveness of the  proposed SRDNet. Since there are too few images in these datasets for deep learning algorithm, we augment the training data. With respect to the specific details, please see the implementation details section.**

**Moreover, The code about data pre-processing in SRDNet (https://github.com/LTTdouble/SRDNet) folder [data pre-processing] or ( https://github.com/qianngli/MCNet/tree/master/data_pre-processing "data pre-processing"). The folder contains three parts, including training set augment, test set pre-processing, and band mean for all training set.**


Requirement
**python 3.7, Pytorch=1.7.1, cuda 11.0, RTX 3090 GPU**

**After setting the options and dataset paths, you can directly run the training or testing process.**

 python train.py

 python test.py


