#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/python.hpp>

#include "conversion.h"
#include "brisk/brisk.h"

namespace py = boost::python;

static cv::Mat get_gray_img(PyObject *p_img);
static std::vector<cv::KeyPoint> detect(cv::Mat img, PyObject *p_thresh, PyObject *p_octaves);
static PyObject* keypoints_ctopy(std::vector<cv::KeyPoint> keypoints);

static cv::Mat get_gray_img(PyObject *p_img) {
    // get image and convert if necessary
    NDArrayConverter cvt;
    cv::Mat img_temp = cvt.toMat(p_img);
    cv::Mat img;
    if (img_temp.channels() == 1) {
        img = img_temp;
    } else {
        cv::cvtColor(img_temp, img, CV_BGR2GRAY);
    }

    return img;
}

static std::vector<cv::KeyPoint> detect(cv::Mat img, PyObject *p_thresh, PyObject *p_octaves) {
    // parse python arguments
    int thresh;
    int octaves;
    PyArg_Parse(p_thresh, "i", &thresh);
    PyArg_Parse(p_octaves, "i", &octaves);

    // detect keypoints
    cv::Ptr<cv::FeatureDetector> detector;
    detector = new brisk::BriskFeatureDetector(thresh, octaves);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img, keypoints);

    return keypoints;
}

static PyObject* keypoints_ctopy(std::vector<cv::KeyPoint> keypoints) {
    size_t num_keypoints = keypoints.size();
    PyObject* ret_keypoints = PyList_New(num_keypoints);
    // import cv2
    PyObject* cv2_mod = PyImport_ImportModule("cv2");

    for(size_t i = 0; i < num_keypoints; ++i) {
        // cv2_keypoint = cv2.KeyPoint()
        // TODO: PyInstance_New is maybe better
        // cv2_keypoint has maybe a memory leak
        PyObject* cv2_keypoint = PyObject_CallMethod(cv2_mod, "KeyPoint", "");

        // build values
        PyObject* cv2_keypoint_size = Py_BuildValue("f", keypoints[i].size);
        PyObject* cv2_keypoint_angle = Py_BuildValue("f", keypoints[i].angle);
        PyObject* cv2_keypoint_response = Py_BuildValue("f", keypoints[i].response);
        PyObject* cv2_keypoint_pt_x = Py_BuildValue("f", keypoints[i].pt.x);
        PyObject* cv2_keypoint_pt_y = Py_BuildValue("f", keypoints[i].pt.y);

        // pack into tuple
        PyObject* cv2_keypoint_pt = PyTuple_New(2);
        PyTuple_SetItem(cv2_keypoint_pt, 0, cv2_keypoint_pt_x);
        PyTuple_SetItem(cv2_keypoint_pt, 1, cv2_keypoint_pt_y);

        // set attributes
        PyObject_SetAttrString(cv2_keypoint, "size", cv2_keypoint_size);
        PyObject_SetAttrString(cv2_keypoint, "angle", cv2_keypoint_angle);
        PyObject_SetAttrString(cv2_keypoint, "response", cv2_keypoint_response);
        PyObject_SetAttrString(cv2_keypoint, "pt", cv2_keypoint_pt);

        // add keypoint to list
        PyList_SetItem(ret_keypoints, i, cv2_keypoint);

        Py_DECREF(cv2_keypoint_size);
        Py_DECREF(cv2_keypoint_angle);
        Py_DECREF(cv2_keypoint_response);
        Py_DECREF(cv2_keypoint_pt_x);
        Py_DECREF(cv2_keypoint_pt_y);
        Py_DECREF(cv2_keypoint_pt);
    }

    Py_DECREF(cv2_mod);

    return ret_keypoints;
}

PyObject* old_detect(PyObject *img, PyObject *thresh_py, PyObject *octaves_py)
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

PyObject* old_compute(PyObject *img, PyObject *keypoints)
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

PyObject* detect_keypoints(PyObject *p_img, PyObject *p_thresh, PyObject *p_octaves) {
    cv::Mat img = get_gray_img(p_img);
    std::vector<cv::KeyPoint> keypoints = detect(img, p_thresh, p_octaves);

    PyObject* ret_keypoints = keypoints_ctopy(keypoints);

    return ret_keypoints;
}

void extract_features(PyObject *p_img, PyObject *p_keypoints) {
    // TODO: implement function
    // get image and conver if necessary
    /*NDArrayConverter cvt;
    cv::Mat img_temp = cvt.toMat(p_img);
    cv::Mat img;
    if (img_temp.channels() == 1) {
        img = img_temp;
    } else {
        cv::cvtColor(img_temp, img, CV_BGR2GRAY);
    }

    Py_ssize_t num_keypoints = PyList_Size(p_keypoints);
    std::vector<cv::KeyPoint> keypoints;
    // import cv2
    PyObject* cv2_mod = PyImport_ImportModule("cv2");*/
}

PyObject* detect_and_extract(PyObject *p_img, PyObject *p_thresh, PyObject *p_octaves) {
    NDArrayConverter cvt;

    cv::Mat img = get_gray_img(p_img);
    std::vector<cv::KeyPoint> keypoints = detect(img, p_thresh, p_octaves);

    cv::Mat descriptors;
    cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
    descriptorExtractor = new brisk::BriskDescriptorExtractor();
    descriptorExtractor->compute(img, keypoints, descriptors);

    PyObject* ret = PyList_New(2);
    PyObject* ret_keypoints = keypoints_ctopy(keypoints);
    PyList_SetItem(ret, 0, cvt.toNDArray(descriptors));
    PyList_SetItem(ret, 1, ret_keypoints);
    // TODO: decrement reference doesn't work
    // Py_DECREF(ret_keypoints);

    return ret;
}

static void init() {
    Py_Initialize();
    import_array();
}

BOOST_PYTHON_MODULE(pybrisk) {
    init();
    py::def("detect", old_detect);
    py::def("compute", old_compute);
    py::def("detect_keypoints", detect_keypoints);
    py::def("extract_features", extract_features);
    py::def("detect_and_extract", detect_and_extract);
}
