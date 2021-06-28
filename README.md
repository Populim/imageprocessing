# Road satellite image detection

Repository for the Digital Image Processing course (SCC0251).

# image-processing
Repo for the Digital Image Processing course taken on 2021.

#### Bruno Baldissera
#### Bruno Gazoni - 7585037
#### Matheus Populim 10734710
#### Rafael Ceneme 9898610

# The main objective of the project

The objective of this project is to identify and segment roads in aerial images of city areas, in particular segmenting images with grassy areas cut by roads.

# The description of input images (with examples) including the source of the images --- website, paper, acquired by the student(s)

The input images have been taken from the following Kaggle dataset:
https://www.kaggle.com/deeenizka/road-segmentation

According to the uploader, the images are “[s]atellite images of Toronto which [he] used for road segmentation, every image is a square 1500x1500 pixels.” We will select some images with particular contrast between grassy and road areas to narrow down the scope of our project into something manageable.

Description of steps to reach the objective, including the methods that are intented to be used (e.g., image enhancement, edge-detection, morphology segmentation, texture analysis, color analysis, keypoint detection etc.). (2 pts)


# First approach

Surveying the literature in road segmentation based on sattelite images (or SAR images) we found that the most recent approaches are based on deep learning methods, varying from encoder-decoder architectures to CNNs and GANs. ([1](https://www.mdpi.com/2072-4292/13/5/1011), [2](https://acadpubl.eu/hub/2018-119-16/2/545.pdf), [3](https://arxiv.org/pdf/2001.05566.pdf))

However, for our purposes we were interested in finding a more simple baseline founded on the notions of convolutional filters, thresholding and colour processing.

We first implemented a simple threshold function in a few test images and saw it as a good direction to follow. Then we tried combining the threshold filter with other methods, looking for progress in the outlining of our region of interest. We performed trials using:

* The manipulation of the saturation of the image using its HSV representation, in order to further differentiate the road pixels from the rest.
* The separation between the shade of gray closer to the roads and another ones (caused by shadows or excessive light) by the V value of HSV image representation. 
* A gamma filter to spread the distribution of the saturation values.
* A border detection filter (first as a standalone test, but we expected to use this a stepping stone, applying in the future some transformation that would better use the border detection for the final goal, maybe a sort of flood fill).


# References

Nachmany, Yoni & Alemohammad, Hamed. (2019). Detecting Roads from Satellite Imagery in the Developing World. 

https://repository.tudelft.nl/islandora/object/uuid:21fc20a8-455d-4583-9698-4fea04516f03/datastream/OBJ2/download


https://www.cs.toronto.edu/~hinton/absps/road_detection.pdf


https://www.sciencedirect.com/science/article/abs/pii/S0303243406000171
