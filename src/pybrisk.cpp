#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/python.hpp>

#include "conversion.h"
#include "brisk/brisk.h"

namespace py = boost::python;

PyObject* detect(PyObject *img, PyObject *thresh_py, PyObject *octaves_py)
{
    NDArrayConverter cvt;
    int thresh;
    int octaves;

    cv::Mat img_rgb = cvt.toMat(img);
    cv::Ptr<cv::FeatureDetector> detector;

    PyArg_Parse(thresh_py, "i", &thresh);
    PyArg_Parse(octaves_py, "i", &octaves);
    detector = new brisk::BriskFeatureDetector(thresh, octaves);

    cv::Mat img_gray;
    if (img_rgb.channels() == 1) {
        img_gray = img_rgb;
    } else {
        cv::cvtColor(img_rgb, img_gray, CV_BGR2GRAY);
    }

    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img_gray, keypoints);

    cv::Mat img_keypoints;

    size_t num_keypoints = keypoints.size();
    // [x, y, size, angle, response]
    cv::Mat ret_keypoints(num_keypoints, 5, cv::DataType<float>::type);

    for(size_t i = 0; i < num_keypoints; ++i)
    {
        ret_keypoints.at<float>(i, 0) = keypoints[i].pt.x;
        ret_keypoints.at<float>(i, 1) = keypoints[i].pt.y;
        ret_keypoints.at<float>(i, 2) = keypoints[i].size;
        ret_keypoints.at<float>(i, 3) = keypoints[i].angle;
        ret_keypoints.at<float>(i, 4) = keypoints[i].response;
    }

    PyObject* ret = cvt.toNDArray(ret_keypoints);
    return ret;
}

PyObject* compute(PyObject *img, PyObject *keypoints)
{
    NDArrayConverter cvt;

    cv::Mat img_rgb = cvt.toMat(img);
    cv::Mat img_gray;
    if (img_rgb.channels() == 1) {
        img_gray = img_rgb;
    } else {
        cv::cvtColor(img_rgb, img_gray, CV_BGR2GRAY);
    }

    cv::Mat keypoints_array = cvt.toMat(keypoints);
    PyObject* ret = PyList_New(2);
    std::vector<cv::KeyPoint> keypoints_cv;

    size_t num_keypoints = keypoints_array.rows;

    for(size_t i = 0; i < num_keypoints; ++i)
    {
        cv::KeyPoint temp;
        temp.pt.x = keypoints_array.at<float>(i, 0);
        temp.pt.y = keypoints_array.at<float>(i, 1);
        temp.size = keypoints_array.at<float>(i, 2);
        temp.angle = keypoints_array.at<float>(i, 3);
        temp.response = keypoints_array.at<float>(i, 4);
        keypoints_cv.push_back(temp);
    }

    cv::Mat descriptors;
    cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
    descriptorExtractor = new brisk::BriskDescriptorExtractor();

    descriptorExtractor->compute(img_gray, keypoints_cv, descriptors);

    num_keypoints = keypoints_cv.size();
    cv::Mat ret_keypoints(num_keypoints, 5, cv::DataType<float>::type);

    for(size_t i = 0; i < num_keypoints; ++i)
    {
        ret_keypoints.at<float>(i, 0) = keypoints_cv[i].pt.x;
        ret_keypoints.at<float>(i, 1) = keypoints_cv[i].pt.y;
        ret_keypoints.at<float>(i, 2) = keypoints_cv[i].size;
        ret_keypoints.at<float>(i, 3) = keypoints_cv[i].angle;
        ret_keypoints.at<float>(i, 4) = keypoints_cv[i].response;
    }

    PyList_SetItem(ret, 0, cvt.toNDArray(descriptors));
    PyList_SetItem(ret, 1, cvt.toNDArray(ret_keypoints));

    return ret;
}

static void init()
{
    Py_Initialize();
    import_array();
}

BOOST_PYTHON_MODULE(pybrisk)
{
    init();
    py::def("detect", detect);
    py::def("compute", compute);
}
