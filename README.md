# brisk
Effective and efficient generation of keypoints from an image is a well-studied
problem in the literature and forms the basis of numerous Computer Vision
applications. Established leaders in the field are the SIFT and SURF algorithms
which exhibit great performance under a variety of image transformations, with
SURF in particular considered as the most computationally efficient amongst the
high-performance methods to date.

Since the OpenCV brisk-detector is broken
[Bug #2491](http://code.opencv.org/issues/2491 "Bugreport") and the
[original code](http://www.asl.ethz.ch/people/lestefan/personal/BRISK "Original Code")
can't be compiled with new OpenCV versions (conflicting names), I've forked the
original code and made some minor tweaks and a new python wrapper.

# Original Header
> BRISK package: Source Code Release v0.0
> Copyright 2011 Autonomous Systems Lab (ASL), ETH Zurich
> Stefan Leutenegger, Simon Lynen and Margarita Chli
>
> License: BSD (see license file included in this folder)
>
> This software is an implementation of [1]:  
> [1] Stefan Leutenegger, Margarita Chli and Roland Siegwart, BRISK:  
>    Binary Robust Invariant Scalable Keypoints, in Proceedings of the  
>    IEEE International Conference on Computer Vision (ICCV2011).

# Dependencies
## C++
* OpenCV
* cmake

## Python
* Python (dev)
* Numpy

# Building
Download the source code
```bash
cd ${brisk-dir}
wget https://github.com/clemenscorny/brisk/archive/master.zip
unzip master.zip
cd brisk-master
```
Create the Makifile with cmake
```bash
mkdir build
cd build
cmake ..
# or
# cmake .. -DBUILD_PYTHON_BINDING=OFF
# to build without python wrapper
```
Build the project
```bash
make
```
The built files files are stored in `${brisk-dir}/brisk-master/build/bin` and `${brisk-dir}/brisk-master/build/lib`

# Testing
Navigate to the `bin` folder and run the demo
```bash
cd ${brisk-dir}/brisk-master/build/bin/
./demo
```

# Example
## Python
Add the python wrapper folder to the python path
```bash
export PYTHONPATH=${brisk-dir}/brisk-master/build/lib:$PYTHONPATH
```
and run
```python
import brisk
import cv2

img = cv2.imread('path/to/an/image.png')

b = brisk.Brisk()
# detector
kpts = b.detect(img)
# desriptor
kpts, features = b.compute(img, kpts)

# draw and show keypoints in the image
cv2.imshow('Brisk', cv2.drawKeypoints(img, kpts, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
cv2.waitKey()
```

## C++
Take a look at [demo.cpp](/src/demo.cpp "demo.cpp")

**Important note:** This brisk implementation (original implementation) doesn't
calculate the angles in `brisk::BriskFeatureDetector::detect`.
Use this piece of code to compute it
```cpp
cv::Mat img; // use for example cv::imread to load an image
cv::Ptr<cv::FeatureDetector> detector;
detector = new brisk::BriskFeatureDetector(60, 4);
std::vector<cv::KeyPoint> keypoints;
detector->detect(img, keypoints);

// keypoints have invalid angles
// keypoints[i].angle is every time -1

// code of interest
// brisk::BriskDescriptorExtractor::BriskDescriptorExtractor needs some computation
// time. (But you don't have to create for every new image a new new instance of
// this class).
brisk::BriskDescriptorExtractor* descriptor_extractor = new brisk::BriskDescriptorExtractor();
descriptor_extractor->computeAngles(img, keypoints);

// keypoints have valid angles
// keypoints[i].angle works
```
