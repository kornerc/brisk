#include <Python.h>

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include "conversion.h"
#include "brisk/brisk.h"

#include <sys/time.h>

// TODO: add documentation

static cv::Mat get_gray_img(PyObject *p_img);
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

static PyObject* keypoints_ctopy(std::vector<cv::KeyPoint> keypoints) {
    size_t num_keypoints = keypoints.size();
    PyObject* ret_keypoints = PyList_New(num_keypoints);
    // import cv2
    PyObject* cv2_mod = PyImport_ImportModule("cv2");
    PyObject* cv2_keypoint_class = PyObject_GetAttrString(cv2_mod, "KeyPoint");

    for(size_t i = 0; i < num_keypoints; ++i) {
        // cv2_keypoint = cv2.KeyPoint()
        PyObject* cv2_keypoint = PyObject_CallObject(cv2_keypoint_class, NULL);

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
        Py_DECREF(cv2_keypoint_pt);
    }

    Py_DECREF(cv2_mod);
    Py_DECREF(cv2_keypoint_class);

    return ret_keypoints;
}

static PyObject* create(PyObject* self, PyObject* args) {
    brisk::BriskDescriptorExtractor* descriptor_extractor = new brisk::BriskDescriptorExtractor();
    return PyCObject_FromVoidPtr(static_cast<void*>(descriptor_extractor), NULL);
}

static PyObject* destroy(PyObject* self, PyObject* args) {
    PyObject* p_descriptor_extractor;
    if(!PyArg_ParseTuple(args, "O", &p_descriptor_extractor)) {
        return NULL;
    }

    brisk::BriskDescriptorExtractor* descriptor_extractor =
            static_cast<brisk::BriskDescriptorExtractor*>(PyCObject_AsVoidPtr(p_descriptor_extractor));
    delete descriptor_extractor;

    Py_RETURN_NONE;
}

static PyObject* detect(PyObject* self, PyObject* args) {
    PyObject* p_descriptor_extractor;
    PyObject* p_img;
    int thresh;
    int octaves;
    if(!PyArg_ParseTuple(args, "OOii", &p_descriptor_extractor, &p_img,
            &thresh, &octaves)) {
        return NULL;
    }

    cv::Mat img = get_gray_img(p_img);

    // detect keypoints
    cv::Ptr<cv::FeatureDetector> detector;
    detector = new brisk::BriskFeatureDetector(thresh, octaves);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img, keypoints);

    brisk::BriskDescriptorExtractor* descriptor_extractor =
            static_cast<brisk::BriskDescriptorExtractor*>(PyCObject_AsVoidPtr(p_descriptor_extractor));
    descriptor_extractor->computeAngles(img, keypoints);

    PyObject* ret_keypoints = keypoints_ctopy(keypoints);

    return ret_keypoints;
}

static PyObject* compute(PyObject* self, PyObject* args) {
    PyObject* p_descriptor_extractor;
    PyObject *p_img;
    PyObject *p_keypoints;
    if(!PyArg_ParseTuple(args, "OOO", &p_descriptor_extractor, &p_img,
            &p_keypoints)) {
        return NULL;
    }

    cv::Mat img = get_gray_img(p_img);
    Py_ssize_t num_keypoints = PyList_Size(p_keypoints);

    std::vector<cv::KeyPoint> keypoints;

    for(Py_ssize_t i = 0; i < num_keypoints; ++i) {
        keypoints.push_back(cv::KeyPoint());
        PyObject* cv2_keypoint = PyList_GetItem(p_keypoints, i);

        // get attributes
        PyObject* cv2_keypoint_size = PyObject_GetAttrString(cv2_keypoint, "size");
        PyObject* cv2_keypoint_angle = PyObject_GetAttrString(cv2_keypoint, "angle");
        PyObject* cv2_keypoint_response = PyObject_GetAttrString(cv2_keypoint, "response");
        PyObject* cv2_keypoint_pt = PyObject_GetAttrString(cv2_keypoint, "pt");
        PyObject* cv2_keypoint_pt_x = PyTuple_GetItem(cv2_keypoint_pt, 0);
        PyObject* cv2_keypoint_pt_y = PyTuple_GetItem(cv2_keypoint_pt, 1);

        // set data
        PyArg_Parse(cv2_keypoint_size, "f", &keypoints[i].size);
        PyArg_Parse(cv2_keypoint_angle, "f", &keypoints[i].angle);
        PyArg_Parse(cv2_keypoint_response, "f", &keypoints[i].response);
        PyArg_Parse(cv2_keypoint_pt_x, "f", &keypoints[i].pt.x);
        PyArg_Parse(cv2_keypoint_pt_y, "f", &keypoints[i].pt.y);

        Py_DECREF(cv2_keypoint_size);
        Py_DECREF(cv2_keypoint_angle);
        Py_DECREF(cv2_keypoint_response);
        Py_DECREF(cv2_keypoint_pt);
    }

    cv::Mat descriptors;
    brisk::BriskDescriptorExtractor* descriptor_extractor =
            static_cast<brisk::BriskDescriptorExtractor*>(PyCObject_AsVoidPtr(p_descriptor_extractor));
    descriptor_extractor->compute(img, keypoints, descriptors);

    NDArrayConverter cvt;
    PyObject* ret = PyTuple_New(2);
    PyTuple_SetItem(ret, 0, keypoints_ctopy(keypoints));
    PyTuple_SetItem(ret, 1, cvt.toNDArray(descriptors));

    return ret;
}

static PyMethodDef brisk_methods[] = {
     {"create", create,  METH_NOARGS, ""},
     {"destroy", destroy,  METH_VARARGS, ""},
     {"detect", detect,  METH_VARARGS, ""},
     {"compute", compute,  METH_VARARGS, ""},
     {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initpybrisk() {
    Py_InitModule("pybrisk", brisk_methods);
}
