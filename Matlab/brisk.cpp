/*=================================================================
 *
 * BRISK.C  .MEX interface to the BRISK C++ library
 *	    Detects, extracts and matches BRISK features
 *          Implementation according to 
 *
 *      [1] Stefan Leutenegger, 
 *          Margarita Chli and Roland Siegwart, BRISK: Binary 
 *          Robust Invariant Scalable Keypoints, in Proceedings of 
 *          the IEEE International Conference on Computer Vision 
 *          (ICCV) 2011.
 *
 * The calling syntax is:
 *
 *	varargout = brisk(subfunction, morevarargin)
 *
 *      where subfunction is to be used in order:
 *
 *      'init'        Initialize brisk. Optionally pass arguments to 
 *                    set properties (see below). 
 *                    Attention: this will create the pattern look-up table,
 *                    so this may take some fraction of a second. 
 *                    Do not rerun!
 *
 *      'set'         Set properties. The following may be set:
 *                    '-threshold'    FAST/AGAST detection threshold.
 *                                  The default value is 60.
 *                    '-octaves'      No. octaves for the detection.
 *                                  The default value is 4.
 *                    '-patternScale' Scale factor for the BRISK pattern.
 *                                  The default value is 1.0.
 *                    '-type'         BRISK special type 'S', 'U', 'SU'.
 *                                  By default, the standard BRISK is used.
 *                                    See [1] for explanations on this.
 *                    Attention: if the patternScale or the type is reset, 
 *                    the pattern will be regenerated, which is time-
 *                    consuming!
 *
 *      'loadImage'   Load an image either from Matlab workspace by passing
 *                    a UINT8 Matrix as a second argument, or by specifying 
 *                    a path to an image:
 *                        brisk('loadImage',imread('path/to/image'));
 *                        brisk('loadImage','path/to/image');
 *
 *      'detect'      Detect the keypoints. Optionally get the points back:
 *                      brisk('detect');
 *                      keyPoints=brisk('detect');
 *
 *      'describe'    Get the descriptors and the corresponding keypoints
 *                      [keyPoints,descriptors]=brisk('detect');
 *
 *      'radiusMatch' Radius match.
 *                      [indicesOfSecondKeyPoints]=brisk('radiusMatch',...
 *                          firstKeypoints,secondKeyPoints);
 * 
 *      'knnMatch'    k-nearest neighbor match.
 *                      [indicesOfSecondKeyPoints]=brisk('knnMatch',...
 *                          firstKeypoints,secondKeyPoints,k);
 *
 *      'image'       Returns the currently used gray-scale image
 *                      image=brisk('image');
 *
 *      'terminate'   Free the memory.
 *
 *
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

#include "brisk_interface.h"

// helper function to test input args
bool mxIsScalarNonComplexDouble(const mxArray* arg){
    mwSize mrows,ncols;
    mrows = mxGetM(arg);
    ncols = mxGetN(arg);
    if( !mxIsDouble(arg) || mxIsComplex(arg) ||
      !(mrows==1 && ncols==1) )
        return false;
    return true;
}

// constructor: set pointers to 0
BriskInterface::BriskInterface( int nlhs, mxArray *plhs[], 
            int nrhs, const mxArray *prhs[] ){
    // init default parameters
    threshold = 60;
    octaves = 4;
    rotationInvariant=true;
    scaleInvariant=true; 
    patternScale=1.0f;
    
    // init the pointers
    p_detector=0;
    p_descriptor=0;
    p_matcher=new cv::BruteForceMatcher<cv::HammingSse>;
    
    // use the inputs to set
    set(nlhs, plhs, nrhs, prhs);
}

BriskInterface::~BriskInterface(){
    // cleanup pointers
    if(p_detector!=0) delete p_detector;
    if(p_descriptor!=0) delete p_descriptor;
    if(p_matcher!=0) delete p_matcher;
}

// reset octaves / threshold / Pattern
inline void BriskInterface::set(int nlhs, mxArray *plhs[], 
            int nrhs, const mxArray *prhs[]){
    if((nrhs-1)%2!=0) mexErrMsgTxt("Bad input.");
    
    // remember what to re-initialize
    bool initDetector=false;
    bool initDescriptor=false;
    bool scaleSet=false;
    
    for(int i=1; i<nrhs; i+=2){
        // parse option
        char* str2=mxArrayToString(prhs[i]);
        if(!(mxGetClassID(prhs[i])==mxCHAR_CLASS))
            mexErrMsgTxt("Bad input.");
        if(strcmp(str2,"threshold")==0){
            if(!mxIsScalarNonComplexDouble(prhs[i+1]))
                mexErrMsgTxt("Bad input.");
            double* x=mxGetPr(prhs[i+1]);
            // bound
            if(*x<0) threshold =0;
            else if(*x>255) threshold =255;
            else threshold=int(*x);
            initDetector=true;
        }
        else if(strcmp(str2,"octaves")==0){
            if(!mxIsScalarNonComplexDouble(prhs[i+1]))
                mexErrMsgTxt("Bad input.");
            double* x=mxGetPr(prhs[i+1]);
            if(*x<0) octaves=0;
            else octaves=int(*x);
            initDetector=true;
        }
        else if(strcmp(str2,"patternScale")==0){
            if(!mxIsScalarNonComplexDouble(prhs[i+1]))
                mexErrMsgTxt("Bad input.");
            double* x=mxGetPr(prhs[i+1]);
            if(*x<0) patternScale=1.0;
            else patternScale=*x;
            scaleSet=true;
            initDescriptor=true;
        }
        else if(strcmp(str2,"type")==0){
            if(!(mxGetClassID(prhs[i+1])==mxCHAR_CLASS))
                mexErrMsgTxt("Bad input.");
            char* str3=mxArrayToString(prhs[i+1]);
            if(strcmp(str3,"S")==0){
                scaleInvariant=false;
                if(scaleSet) patternScale=1.2f; // match brief
            }
            else if (strcmp(str3,"U")==0){
                rotationInvariant=false;
            }
            else if (strcmp(str3,"SU")==0){
                rotationInvariant=false;
                if(scaleSet) patternScale=1.2f; // match brief
            }
            else mexErrMsgTxt("Unrecognized type.");
            initDescriptor=true;
        }
        else mexErrMsgTxt("Unrecognized input option.");
    }
    // reset if requested
    if(initDetector||p_detector==0){
        if(p_detector!=0)
            delete p_detector;
        p_detector=new cv::BriskFeatureDetector(threshold, octaves);
    }
    if(initDescriptor||p_descriptor==0){
        if(p_descriptor!=0)
            delete p_descriptor;
        p_descriptor=new cv::BriskDescriptorExtractor(
                rotationInvariant, scaleInvariant, patternScale);
    }
}

// load an image
inline void BriskInterface::loadImage( int nlhs, mxArray *plhs[], 
            int nrhs, const mxArray *prhs[] ){
    if(nrhs<2) 
        mexErrMsgTxt("No image passed.");
    if((mxGetClassID(prhs[1])==mxUINT8_CLASS)){
        // image dimensions
        int M=mxGetM(prhs[1]);
        int N=mxGetN(prhs[1]);
        mwSize dim=mxGetNumberOfDimensions(prhs[1]);
        if(dim==3){
            // this means we need to merge the channels.
            uchar *data = (uchar*) mxGetData(prhs[1]);
            std::vector<cv::Mat> BGR;
            BGR.push_back(cv::Mat(N/3, M, CV_8U, data+2*N*M/3));
            BGR.push_back(cv::Mat(N/3, M, CV_8U, data+N*M/3));
            BGR.push_back(cv::Mat(N/3, M, CV_8U, data));

            // merge into one BGR matrix
            cv::Mat imageBGR;
            cv::merge(BGR,imageBGR);
            // color conversion
            cv::cvtColor(imageBGR,img, CV_BGR2GRAY);
            
            // transpose
            img=img.t();
        }
        else if(dim==2){// cast image to a cv::Mat
            uchar* data = (uchar*) mxGetData(prhs[1]); 
            img=cv::Mat(N, M, CV_8U, data);
            
            // transpose
            img=img.t();
        }
        else mexErrMsgTxt("Image dimension must be 2 or 3.");
    }
    else if((mxGetClassID(prhs[1])==mxCHAR_CLASS)){
        char* fname = mxArrayToString(prhs[1]);
        img=cv::imread(fname,0); // forcing gray
        if(img.data==0){
            mexPrintf("%s ",fname);
            mexErrMsgTxt("Image could not be loaded.");
        }
        //mexMakeMemoryPersistent(&img);
    }
    else
        mexErrMsgTxt("Pass an UINT8_T image matrix or a path.");
}

// detection
inline void BriskInterface::detect( int nlhs, mxArray *plhs[], 
            int nrhs, const mxArray *prhs[] ){
    if(img.empty()) 
            mexErrMsgTxt("Currently no image loaded.");
        
    // actual detection step
    assert(p_detector);
    p_detector->detect(img, keypoints);

    // send the keypoints to the user, if he wants it
    // allocate plhs
    if(nlhs>=1){
        const int keypoint_size=keypoints.size();
        mxArray* tmp;
        tmp=mxCreateDoubleMatrix(4,keypoint_size,mxREAL);
        double *ptr=mxGetPr(tmp);
        // fill it - attention: in Matlab, memory is transposed...
        for(int k=0; k<keypoint_size; k++){
            const int k4=4*k;
            ptr[k4]=keypoints[k].pt.x;
            ptr[k4+1]=keypoints[k].pt.y;
            ptr[k4+2]=keypoints[k].size;
            ptr[k4+3]=-1;
        }
        
        // finally, re-transpose for better readibility:
        mexCallMATLAB(1, plhs, 1, &tmp, "transpose");
    }   
}
// descriptor extraction
inline void BriskInterface::describe(int nlhs, mxArray *plhs[], 
            int nrhs, const mxArray *prhs[]){
    // in this case, the user is forced to pass two lhs args
    if(nlhs!=2) 
        mexErrMsgTxt("Two left-hand side arguments must be passed.");
    if(img.empty())
        mexErrMsgTxt("No image loaded.");

    // check the keypoints
    if(keypoints.size()==0)
        mexErrMsgTxt("Keypoints empty. Run detect.");

    // now we can extract the descriptors
    cv::Mat descriptors;
    assert(p_descriptor);
    p_descriptor->compute(img,keypoints,descriptors);

    // allocate the lhs descriptor matrix
    int dim[2];
    dim[0]=p_descriptor->descriptorSize();
    dim[1]=keypoints.size();
    mxArray* tmp1=mxCreateNumericArray(2,dim,mxUINT8_CLASS,mxREAL);
    uchar* data = (uchar*) mxGetData(tmp1); 
    // copy - kind of dumb, but necessary due to the matlab memory 
    // management
    memcpy(data,descriptors.data,dim[0]*dim[1]);
    // transpose for better readibility
    mexCallMATLAB(1, &plhs[1], 1, &tmp1, "transpose");

    // also write the keypoints
    const int keypoint_size=keypoints.size();
    mxArray* tmp;
    tmp=mxCreateDoubleMatrix(4,keypoint_size,mxREAL);
    double *ptr=mxGetPr(tmp);
    // fill it - attention: in Matlab, memory is transposed...
    for(int k=0; k<keypoint_size; k++){
        const int k4=4*k;
        ptr[k4]=keypoints[k].pt.x;
        ptr[k4+1]=keypoints[k].pt.y;
        ptr[k4+2]=keypoints[k].size;
        ptr[k4+3]=keypoints[k].angle;
    }
    
    // finally, re-transpose for better readibility:
    mexCallMATLAB(1, plhs, 1, &tmp, "transpose");
}
    
// descriptor matching
inline void BriskInterface::radiusMatch( int nlhs, mxArray *plhs[], 
            int nrhs, const mxArray *prhs[] ){
    // insure correct arguments
    if(nrhs<3)
        mexErrMsgTxt("Two descriptors must be passed.");
    if(nlhs!=1)
        mexErrMsgTxt("Specify one output argument.");
    if(mxGetClassID(prhs[1])!=mxUINT8_CLASS || 
            mxGetClassID(prhs[2])!=mxUINT8_CLASS)
        mexErrMsgTxt("Wrong descriptor type.");

    // get the two input descriptors
    mxArray *d1, *d2;
    mexCallMATLAB(1, &d1, 1, (mxArray**)&prhs[1], "transpose");
    mexCallMATLAB(1, &d2, 1, (mxArray**)&prhs[2], "transpose");
    // cast to cv::Mat
    const int N1 = mxGetN(d1);
    const int N2 = mxGetN(d2);
    const int M = mxGetM(d1);
    if(M!=mxGetM(d2))
        mexErrMsgTxt("Incompatible descriptors (wrong no. bytes).");
    uchar* data1 = (uchar*)mxGetData(d1);
    uchar* data2 = (uchar*)mxGetData(d2);
    cv::Mat d1m(N1,M,CV_8U,data1);
    cv::Mat d2m(N2,M,CV_8U,data2);

    // get the radius, if provided
    int k=90;
    if(nrhs>3){
        if(!mxIsScalarNonComplexDouble(prhs[3]))
            mexErrMsgTxt("Wrong type for radius.");
        double* kd=mxGetPr(prhs[3]);
        if (*kd<0.0) *kd=0.0;
        k=int(*kd);
    }

    // perform the match
    std::vector<std::vector<cv::DMatch> > matches;
    p_matcher->radiusMatch(d1m,d2m,matches,k);

    // assign the output - first determine the matrix size
    const unsigned int msize=matches.size();
    unsigned int maxMatches=0;
    for(int m=0; m<msize; m++){
        const unsigned int size=matches[m].size();
        if(size>maxMatches) maxMatches=size;
    }

    // allocate memory
    plhs[0]=mxCreateDoubleMatrix(msize,maxMatches,mxREAL);
    double *data=mxGetPr(plhs[0]);

    // fill
    for(int m=0; m<msize; m++){
        const unsigned int size=matches[m].size();
        for(int s=0; s<size; s++){
            data[m+s*msize]=matches[m][s].trainIdx+1;
        }
    }
}
inline void BriskInterface::knnMatch( int nlhs, mxArray *plhs[], 
        int nrhs, const mxArray *prhs[] ){
    // insure correct arguments
    if(nrhs<3)
        mexErrMsgTxt("Two descriptors must be passed.");
    if(nlhs!=1)
        mexErrMsgTxt("Specify one output argument.");
    if(mxGetClassID(prhs[1])!=mxUINT8_CLASS || 
            mxGetClassID(prhs[2])!=mxUINT8_CLASS)
        mexErrMsgTxt("Wrong descriptor type.");

    // get the two input descriptors
    mxArray *d1, *d2;
    mexCallMATLAB(1, &d1, 1, (mxArray**)&prhs[1], "transpose");
    mexCallMATLAB(1, &d2, 1, (mxArray**)&prhs[2], "transpose");
    // cast to cv::Mat
    const int N1 = mxGetN(d1);
    const int N2 = mxGetN(d2);
    const int M = mxGetM(d1);
    if(M!=mxGetM(d2))
        mexErrMsgTxt("Incompatible descriptors (wrong no. bytes).");
    uchar* data1 = (uchar*)mxGetData(d1);
    uchar* data2 = (uchar*)mxGetData(d2);
    cv::Mat d1m(N1,M,CV_8U,data1);
    cv::Mat d2m(N2,M,CV_8U,data2);

    // get the number of nearest neighbors if provided
    int k=1;
    if(nrhs>3){
        if(!mxIsScalarNonComplexDouble(prhs[3]))
            mexErrMsgTxt("Wrong type for no. nearest neighbors.");
        double* kd=mxGetPr(prhs[3]);
        if (*kd<1.0) *kd=1.0;
        k=int(*kd);
    }

    // perform the match
    std::vector<std::vector<cv::DMatch> > matches;
    p_matcher->knnMatch(d1m,d2m,matches,k);

    // assign the output - first determine the matrix size
    const unsigned int msize=matches.size();
    const unsigned int maxMatches=k;

    // allocate memory
    plhs[0]=mxCreateDoubleMatrix(msize,maxMatches,mxREAL);
    double *data=mxGetPr(plhs[0]);

    // fill
    for(int m=0; m<msize; m++){
        const unsigned int size=matches[m].size();
        for(int s=0; s<size; s++){
            data[m+s*msize]=matches[m][s].trainIdx+1;
        }
    }
}
    
// grayImage access
inline void BriskInterface::image( int nlhs, mxArray *plhs[], 
        int nrhs, const mxArray *prhs[] ){
    if(nlhs!=1)
        mexErrMsgTxt("No output variable specified.");
    if(nrhs!=1)
        mexErrMsgTxt("bad input.");
    if(img.empty())
        mexErrMsgTxt("No image loaded.");
    int dim[2];

    // depending on whether or not the image was imported from Matlab 
    // workspace, it needs to be transposed or not
    // must be transposed
    dim[0]=img.cols;
    dim[1]=img.rows;
    mxArray* tmp=mxCreateNumericArray(2,dim,mxUINT8_CLASS,mxREAL);
    uchar* dst=(uchar*)mxGetData(tmp);
    memcpy(dst,img.data,img.cols*img.rows);
    mexCallMATLAB(1, plhs, 1, &tmp, "transpose");
}


// the interface object
BriskInterface* p_briskInterface=0;

// this is the actual (single) entry point:
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )     
{
    // user must provide at least one right-hand argument:
	if(nrhs < 1) mexErrMsgTxt("No input specified.");
    // and this must be a string
    if(!(mxGetClassID(prhs[0])==mxCHAR_CLASS)) mexErrMsgTxt("Bad input.");
    // parse the first input argument:
    char* str=mxArrayToString(prhs[0]);
    if(strcmp(str,"init")==0) {
        if(!p_briskInterface) {
            p_briskInterface=new BriskInterface(nlhs, plhs, nrhs, prhs);
            // make sure the memory requested persists. 
            //mexMakeMemoryPersistent(p_briskInterface);
        }
        else{
            mexErrMsgTxt("Brisk is already initialized.");
        }
    }
    else if(strcmp(str,"set")==0){
        p_briskInterface->set(nlhs, plhs, nrhs, prhs);
    }
    else if(strcmp(str,"loadImage")==0){
        // init if necessary
        if(!p_briskInterface) {
            p_briskInterface = new BriskInterface(nlhs, plhs, 1, prhs);
            //mexMakeMemoryPersistent(p_briskInterface);
        }
        p_briskInterface->loadImage(nlhs, plhs, nrhs, prhs);
    }
    else if(strcmp(str,"detect")==0){
        // force initialized
        if(!p_briskInterface) {
            mexErrMsgTxt("Not initialized, no image loaded.");
        }
        p_briskInterface->detect(nlhs, plhs, nrhs, prhs);
    }
    else if(strcmp(str,"describe")==0) {
        // force initialized
        if(!p_briskInterface) {
            mexErrMsgTxt("Not initialized, no image loaded.");
        }
        p_briskInterface->describe(nlhs, plhs, nrhs, prhs);
    }
    else if(strcmp(str,"radiusMatch")==0) {
        // init if necessary
        if(!p_briskInterface) {
            p_briskInterface = new BriskInterface(nlhs, plhs, 1, prhs);
            //mexMakeMemoryPersistent(p_briskInterface);
        }
        
        p_briskInterface->radiusMatch(nlhs,plhs,nrhs,prhs);
    }
    else if(strcmp(str,"knnMatch")==0) {
        // init if necessary
        if(!p_briskInterface) {
            p_briskInterface = new BriskInterface(nlhs, plhs, 1, prhs);
            //mexMakeMemoryPersistent(p_briskInterface);
        }
        p_briskInterface->knnMatch(nlhs,plhs,nrhs,prhs);
    }
    else if(strcmp(str,"image")==0) {
        // init if necessary
        if(!p_briskInterface) {
           mexErrMsgTxt("Not initialized, no image loaded.");
        }
        p_briskInterface->image(nlhs,plhs,nrhs,prhs);
    }
    else if(strcmp(str,"terminate")==0) {
        if(p_briskInterface) {
            delete p_briskInterface;
            p_briskInterface=0;
        }
        else{
            mexErrMsgTxt("Brisk was not initialized anyways.");
        }
    }
    else{
        mexErrMsgTxt("Unrecognized input.");
    }
}
