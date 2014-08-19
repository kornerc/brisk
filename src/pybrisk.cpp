#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/python.hpp>

#include "conversion.h"
#include "brisk/brisk.h"

namespace py = boost::python;

PyObject* detect(PyObject *img)
{
    NDArrayConverter cvt;

    cv::Mat img_rgb = cvt.toMat(img);
    cv::Ptr<cv::FeatureDetector> detector;
    detector = new brisk::BriskFeatureDetector(60, 4);

    cv::Mat img_gray;
    cv::cvtColor(img_rgb, img_gray, CV_BGR2GRAY);

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

static void init()
{
    Py_Initialize();
    import_array();
}

BOOST_PYTHON_MODULE(pybrisk)
{
    init();
    py::def("detect", detect);
}
