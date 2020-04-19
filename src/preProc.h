//
//  preProc.h
//  AmericanSignLanguage_AI
//
//  Created by Reza Rajabi on 2020-04-16.
//  Copyright © 2020 RR. All rights reserved.
//

#ifndef preProc_h
#define preProc_h

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


void pre_process_image(const cv::Mat& image, cv::Mat& sample, cv::Mat& singleRow) {
    image.copyTo(sample);
    cv::cvtColor(sample, sample, cv::COLOR_BGR2GRAY);
    cv::resize(sample, sample, cv::Size(28,28)); // sample refrence is set
    cv::Mat copy;
    sample.copyTo(copy);
    copy.convertTo(copy, CV_64F, 1.0); // convert to double in [0,1]
    cv::Mat(1, 784, CV_64F, copy.data).copyTo(singleRow);
}
    

#endif /* preProc_h */
