/**
 */

#import "FaceDetectionHelper.h"
#import "FaceDetection.h"
#import "FaceDetectionEvent.h"
#import <ImageIO/ImageIO.h>
#import <AIRExtHelpers/MPStringUtils.h>
#import <QuartzCore/QuartzCore.h>
#import <UIKit/UIKit.h>

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/pixel.h>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"
#include "FaceAnalyser.h"

using namespace std;
static const int kFaceDetectionAccuracyLow = 0;
static const int kFaceDetectionAccuracyHigh = 1;

static FaceDetectionHelper* airFdSharedInstance = nil;
static inline double radians (double degrees) {return degrees * M_PI/180;}



NSMutableArray *openfacepoints = [NSMutableArray array];
NSMutableArray *aulevels = [NSMutableArray array];
NSMutableArray *facepose = [NSMutableArray array];
NSMutableArray *gazeangle = [NSMutableArray array];



NSString *location = [[NSBundle mainBundle] resourcePath];
bool done = false;
int f_n = -1;
int curr_img = -1;
bool openFaceReady = false;
bool faceFound = 0;
string locationTest = "";
int frame_count = 0;
double time_stamp = 0;
int64 t_initial = cv::getTickCount();
LandmarkDetector::FaceModelParameters det_parameters;
LandmarkDetector::CLNF clnf_model;

string au_loc = [location UTF8String] + std::string("/openface_models/AU_predictors/AU_all_best.txt");
string tri_loc = [location UTF8String] + std::string("/openface_models/model/tris_68_full.txt");
FaceAnalysis::FaceAnalyser face_analyser(vector<cv::Vec3d>(), 0.7, 112, 112, au_loc, tri_loc);

BOOL faceSaveImage = false;
BOOL faceDetectAUs = true;
BOOL faceDetectPose = true;
BOOL faceDetectGaze = true;


@implementation FaceDetectionHelper

+ (nonnull id) sharedInstance {
    if( airFdSharedInstance == nil ) {
        airFdSharedInstance = [[FaceDetectionHelper alloc] init];
        
        NSString *location = [[NSBundle mainBundle] resourcePath];
        det_parameters.init();
        det_parameters.model_location = [location UTF8String] + std::string("/openface_models/model/main_clnf_general.txt");
        det_parameters.face_detector_location = [location UTF8String] + std::string("/openface_models/classifiers/haarcascade_frontalface_alt.xml");
        locationTest = [location UTF8String] + std::string("/openface_models/classifiers/haarcascade_frontalface_alt.xml");
        
        
        
        clnf_model.model_location_clnf = [location UTF8String] + std::string("/openface_models/model/main_clnf_general.txt");
        clnf_model.face_detector_location_clnf = [location UTF8String] + std::string("/openface_models/classifiers/haarcascade_frontalface_alt.xml");
        clnf_model.inits();
        
        time_stamp = (double)frame_count * (1.0 / 30.0);
        
        
        
        
        openFaceReady=true;
    }
    return airFdSharedInstance;
}

void visualise_tracking(cv::Mat& captured_image, cv::Mat_<float>& depth_image, const LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters, int frame_count, double fx, double fy, double cx, double cy)
{
    
    double detection_certainty = face_model.detection_certainty;
    bool detection_success = face_model.detection_success;
    
    double visualisation_boundary = 0.2;
    
    if (detection_certainty < visualisation_boundary)
    {
        LandmarkDetector::Draw(captured_image, face_model);
        
        double vis_certainty = detection_certainty;
        if (vis_certainty > 1)
            vis_certainty = 1;
        if (vis_certainty < -1)
            vis_certainty = -1;
        
        vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);
        
        // A rough heuristic for box around the face width
        int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 192.0);
        
        cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(face_model, fx, fy, cx, cy);
        
        // Draw it in reddish if uncertain, blueish if certain
        LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, fx, fy, cx, cy);
    }
}


-(BOOL) run_FaceAR:(cv::Mat)captured_image frame__:(int)frame_count fx__:(double)fx fy__:(double)fy cx__:(double)cx cy__:(double)cy
{
    // Reading the images
    cv::Mat_<float> depth_image;
    cv::Mat_<uchar> grayscale_image;
    
    if(captured_image.channels() == 3)
    {
        cv::cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);
    }else{
        grayscale_image = captured_image.clone();
    }

    
    // The actual facial landmark detection / tracking
    bool detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, clnf_model, det_parameters);

    if(openfacepoints.count > 0)
    {
        [openfacepoints removeAllObjects];
    }
    if(aulevels.count > 0)
    {
        [aulevels removeAllObjects];
    }
    if(facepose.count > 0)
    {
        [facepose removeAllObjects];
    }
    if(gazeangle.count > 0)
    {
        [gazeangle removeAllObjects];
    }
    
    if(detection_success == true)
    {
        //push back points
        std::vector<cv::Point2d> pointVec = LandmarkDetector::CalculateLandmarks(clnf_model);
        for( int i = 0; i < pointVec.size(); ++i)
        {
            string str = std::to_string(pointVec[i].x);
            string str2 = std::to_string(pointVec[i].y);
            string strSplit = ":";
            string pointString = str+strSplit+str2;
            [openfacepoints addObject:[NSString stringWithUTF8String:pointString.c_str()]];
        }
        
        
        //push back pose
        if(faceDetectPose){
            cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(clnf_model, fx, fy, cx, cy);
    
            std::ostringstream posestream;
            posestream << pose_estimate_to_draw;
            std::string posestreamString = posestream.str();
            [facepose addObject:[NSString stringWithUTF8String:posestreamString.c_str()]];
        }
    }
    double detection_certainty = clnf_model.detection_certainty;
    
    if(faceSaveImage){
        visualise_tracking(captured_image, depth_image, clnf_model, det_parameters, frame_count, fx, fy, cx, cy);
    }
    //////////////////////////////////////////////////////////////////////
    /// gaze EstimateGaze
    ///
    cv::Point3f gazeDirection0(0, 0, -1);
    cv::Point3f gazeDirection1(0, 0, -1);
    if (det_parameters.track_gaze && detection_success && clnf_model.eye_model)
    {
        if(faceDetectGaze){
            FaceAnalysis::EstimateGaze(clnf_model, gazeDirection0, fx, fy, cx, cy, true);
            FaceAnalysis::EstimateGaze(clnf_model, gazeDirection1, fx, fy, cx, cy, false);
            
            std::ostringstream gazeStr;
            gazeStr << gazeDirection0;
            std::string gazeString = gazeStr.str();
            [gazeangle addObject:[NSString stringWithUTF8String:gazeString.c_str()]];
            
        
            if(faceSaveImage){
                FaceAnalysis::DrawGaze(captured_image, clnf_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
            }
        }
        if(faceDetectAUs){
            face_analyser.AddNextFrame(captured_image, clnf_model, time_stamp, false, !det_parameters.quiet_mode);
            getAUs();
        }
    }
    if(faceSaveImage)
    {
        UIImage * Cap = [self UIImageFromCVMat:captured_image];
        UIImageWriteToSavedPhotosAlbum(Cap, nil, nil, nil);
    }
    
    return true;
}



- (void) detectFaces:(FREBitmapData2) bitmap detectSaveImage:(BOOL) detectSaveImage detectAUs:(BOOL) detectAUs detectGaze:(BOOL) detectGaze detectPose:(BOOL) detectPose callbackId:(int) callbackId {
    [FaceDetection log:@"FaceDetectionHelper::detectFaces"];
    
    faceSaveImage = detectSaveImage;
    faceDetectAUs = detectAUs;
    faceDetectGaze = detectGaze;
    faceDetectPose = detectPose;
    
    UIImage * portraitImage = [[UIImage alloc] initWithCGImage: [self getCGImageRefFromFREBitmapData:bitmap]
                                                         scale: 1.0
                                                   orientation: UIImageOrientationUp];
    
    
    
    
    dispatch_async( dispatch_get_global_queue( DISPATCH_QUEUE_PRIORITY_HIGH, 0 ), ^{
        
        
        cv::Mat targetImage = [self cvMatFromUIImage:portraitImage];
        cv::Mat targetFlipped;
        cv::Mat greyMat;
        cv::cvtColor(targetImage, greyMat, cv::COLOR_BGR2GRAY);
        
        rotate_90n(greyMat, targetFlipped, 90);
        float fx, fy, cx, cy;
        cx = 1.0*targetFlipped.cols / 2.0;
        cy = 1.0*targetFlipped.rows / 2.0;
        
        fx = 500 * (targetFlipped.cols / portraitImage.size.width);
        fy = 500 * (targetFlipped.rows / portraitImage.size.height);
        
        fx = (fx + fy) / 2.0;
        fy = fx;
        
        
        NSMutableArray* facesResult = [NSMutableArray array];
        [self run_FaceAR:targetFlipped frame__:frame_count fx__:fx fy__:fy cx__:cx cy__:cy];
        
        
        
        [facesResult addObject:[self getFaceJSON]];
        
        dispatch_async( dispatch_get_main_queue(), ^{
            NSMutableDictionary* response = [NSMutableDictionary dictionary];
            response[@"faces"] = facesResult;
            response[@"callbackId"] = @(callbackId);
            [FaceDetection dispatchEvent:FACE_DETECTION_COMPLETE withMessage:[MPStringUtils getJSONString:response]];
        });
    });
}


- (NSString*) getFaceJSON{
    
    
    
    NSMutableDictionary* json = [NSMutableDictionary dictionary];
    json[@"auValues"] = [[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:aulevels options:NSJSONWritingPrettyPrinted error:nil] encoding:NSUTF8StringEncoding];
    json[@"gazePoints"] = [[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:gazeangle options:NSJSONWritingPrettyPrinted error:nil] encoding:NSUTF8StringEncoding];
    json[@"posePoints"] = [[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:facepose options:NSJSONWritingPrettyPrinted error:nil] encoding:NSUTF8StringEncoding];
    json[@"openfacePoints"] = [[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:openfacepoints options:NSJSONWritingPrettyPrinted error:nil] encoding:NSUTF8StringEncoding];
    json[@"debugMessage"] = @"hello from debug";
    
    return [MPStringUtils getJSONString:json];
}

- (CGImageRef) getCGImageRefFromFREBitmapData:(FREBitmapData2) bitmapData {
    size_t width = bitmapData.width;
    size_t height = bitmapData.height;
    
    CGDataProviderRef provider = CGDataProviderCreateWithData( NULL, bitmapData.bits32, (width * height * 4), NULL );
    
    size_t bitsPerComponent = 8;
    size_t bitsPerPixel = 32;
    size_t bytesPerRow = 4 * width;
    CGColorSpaceRef colorSpaceRef = CGColorSpaceCreateDeviceRGB();
    CGBitmapInfo bitmapInfo;
    
    if( bitmapData.hasAlpha ) {
        if( bitmapData.isPremultiplied ) {
            bitmapInfo = kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst;
        } else {
            bitmapInfo = kCGBitmapByteOrder32Little | kCGImageAlphaFirst;
        }
    } else {
        bitmapInfo = kCGBitmapByteOrder32Little | kCGImageAlphaNoneSkipFirst;
    }
    
    CGImageRef imageRef = CGImageCreate( width, height, bitsPerComponent, bitsPerPixel, bytesPerRow, colorSpaceRef, bitmapInfo, provider, NULL, NO, kCGRenderingIntentDefault );
    return imageRef;
}

- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}

void rotate_90n(cv::Mat &src, cv::Mat &dst, int angle)
{
    dst.create(src.size(), src.type());
    if(angle == 270 || angle == -90){
        // Rotate clockwise 270 degrees
        cv::transpose(src, dst);
        cv::flip(dst, dst, 0);
    }else if(angle == 180 || angle == -180){
        // Rotate clockwise 180 degrees
        cv::flip(src, dst, -1);
    }else if(angle == 90 || angle == -270){
        // Rotate clockwise 90 degrees
        cv::transpose(src, dst);
        cv::flip(dst, dst, 1);
    }else if(angle == 360 || angle == 0){
        if(src.data != dst.data){
            src.copyTo(dst);
        }
    }
}

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

//bool reset_FaceAR();
-(BOOL) reset_FaceAR
{
    clnf_model.Reset();
    face_analyser.Reset();
    
    return true;
}

void getAUs(){
    
    auto aus_reg = face_analyser.GetCurrentAUsReg();
    
    vector<string> au_reg_names = face_analyser.GetAURegNames();
    std::sort(au_reg_names.begin(), au_reg_names.end());
    
    // write out ar the correct index
    for (string au_name : au_reg_names)
    {
        for (auto au_reg : aus_reg)
        {
            if (au_name.compare(au_reg.first) == 0)
            {
                //cout << au_name << ", " << au_reg.second << endl;
                std::ostringstream austream;
                austream << au_reg.second;
                std::string austreamString = austream.str();
                std::string comma = ":";
                std::string auvalue = au_name + comma + austreamString;
                [aulevels addObject:[NSString stringWithUTF8String:auvalue.c_str()]];
                //AU_Array.push_back(au_name + comma + str);
                break;
            }
        }
    }
    cout<< "_______" << endl;
    cout<< "" << endl;
    auto aus_class = face_analyser.GetCurrentAUsClass();
    
    vector<string> au_class_names = face_analyser.GetAUClassNames();
    std::sort(au_class_names.begin(), au_class_names.end());
    
    // write out ar the correct index
    for (string au_name : au_class_names)
    {
        for (auto au_class : aus_class)
        {
            if (au_name.compare(au_class.first) == 0)
            {
                //cout << ", " << au_class.second<< endl;
                break;
            }
        }
    }
}

@end
