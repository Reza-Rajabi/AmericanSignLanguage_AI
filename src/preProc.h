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

//#incude "ML_modules/utility.h"


cv::Mat pre_process_image(const cv::Mat& image) {
    cv::Mat copy;
    image.copyTo(copy);
    cv::cvtColor(copy, copy, cv::COLOR_BGR2GRAY);
    cv::resize(copy, copy, cv::Size(28,28));
    copy.convertTo(copy, CV_64F, 1.0/255.0);
    return cv::Mat(1, 784, CV_64F, copy.data);
}
    

#endif /* preProc_h */
