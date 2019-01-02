# VeniceBoat-Classification

This is a simple homework for Machine Learning course at Sapienza University of Rome.

For this homework I've used MarDCT Dataset. You can check this at this link: http://www.diag.uniroma1.it/~labrococo/MAR/index.htm

Please cite this if you use this dataset:
```
@InProceedings{ Bl-Io-Pe-15,
	author = "Bloisi, Domenico D. and Iocchi, Luca and Pennisi, Andrea and Tombolini, Luigi",
	title = {{ARGOS-V}enice Boat Classification},
	booktitle = "Advanced Video and Signal Based Surveillance (AVSS), 2015 12th IEEE International Conference on",
	year = "2015",
	pages = "1--6",
	doi={10.1109/AVSS.2015.7301727}
}
```

## Goals

The aim of this homework is to classify both the kinds of boats that sailed on the Venice Grand Canal and its families too. You can check the first classification on boat-classification.py file, while the second one on family-classification.py.
In the dataset the boats's family are all the folders at depth 1, while the boats's kind are the folders at depth 2. In other words:
```
depth:
0 - Main folder
1 - Family folder
2 - Kind folder
3 - Images
```
It's the same for the test set.

## Tools
I've used two kinds of CNN's, a SmalleVGGNet and a VGG_16. In the end, I have also compared the training results.

Please cite this if you use the VGG_16 net:
```
Very Deep Convolutional Networks for Large-Scale Image Recognition
K. Simonyan, A. Zisserman
arXiv:1409.1556
```

## Authors

* **[Silvio Severino](https://www.linkedin.com/in/silvio-severino-aa2563131/)**
