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

# draw on show keypoints in the image
cv2.imshow('Brisk', cv2.drawKeypoints(img, kpts, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
cv2.waitKey()
```
