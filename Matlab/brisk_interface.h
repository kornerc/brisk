/*=================================================================
 *
 *  BRISK - Binary Robust Invariant Scalable Keypoints
 *  Reference implementation of
 *  [1] Stefan Leutenegger,Margarita Chli and Roland Siegwart, BRISK:
 *  	Binary Robust Invariant Scalable Keypoints, in Proceedings of
 *  	the IEEE International Conference on Computer Vision (ICCV2011).
 *
 * This file is part of BRISK.
 * 
 * Copyright (C) 2011  The Autonomous Systems Lab (ASL), ETH Zurich,
 * Stefan Leutenegger, Simon Lynen and Margarita Chli.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the ASL nor the names of its contributors may be 
 *       used to endorse or promote products derived from this software without 
 *       specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *=================================================================*/

#include <brisk/brisk.h>
#include <mex.h>
#include <vector>
#include <string.h>
#include <opencv2/opencv.hpp>

// helper function to test input args
bool mxIsScalarNonComplexDouble(const mxArray* arg);

class BriskInterface{
public:
    // constructor - calls set
    BriskInterface( int nlhs, mxArray *plhs[], 
            int nrhs, const mxArray *prhs[] );
    // destructor
    ~BriskInterface();
    
    // reset octaves / threshold / pattern
    inline void set(int nlhs, mxArray *plhs[], 
            int nrhs, const mxArray *prhs[]);
    // load an image
    inline void loadImage( int nlhs, mxArray *plhs[], 
            int nrhs, const mxArray *prhs[] );
    // detection
    inline void detect(int nlhs, mxArray *plhs[], 
            int nrhs, const mxArray *prhs[]);
    // descriptor extraction
    inline void describe(int nlhs, mxArray *plhs[], 
            int nrhs, const mxArray *prhs[]);
    // descriptor matching
    inline void radiusMatch( int nlhs, mxArray *plhs[], 
            int nrhs, const mxArray *prhs[] );
    inline void knnMatch( int nlhs, mxArray *plhs[], 
            int nrhs, const mxArray *prhs[] );
    // grayImage access
    inline void image( int nlhs, mxArray *plhs[], 
            int nrhs, const mxArray *prhs[] );
    
        
private:
    cv::BriskFeatureDetector* p_detector; // detector
    cv::BriskDescriptorExtractor* p_descriptor; // descriptor
    cv::BruteForceMatcher<cv::HammingSse>* p_matcher; // matcher
    cv::Mat img; // temporary image stored with loadImage
    std::vector<cv::KeyPoint> keypoints; // temporary keypoint storage
    
    // settings
    unsigned int threshold;
    unsigned int octaves;
    bool rotationInvariant;
    bool scaleInvariant; 
    float patternScale;
};
