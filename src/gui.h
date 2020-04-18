//
//  gui.h
//  AmericanSignLanguage_AI
//
//  Created by Reza Rajabi on 2020-04-17.
//  Copyright © 2020 RR. All rights reserved.
//

#ifndef gui_h
#define gui_h

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>



const int HOLD_INTERVAL = 1000;
enum THEME { BLACK, WHITE, COLOR };
enum POSITION { TOP, BOTT };
enum VISUAL_FLOW { MANUAL, IMEDIATE, STREAM = 33, HOLD = HOLD_INTERVAL, NON = -1 };
const VISUAL_FLOW MODE = HOLD;

const char* window = "ASL.talk";


void writeOnImage(const cv::Mat& image, const char* text, THEME t, POSITION pos) {
    // reset
    cv::Mat copy;
    image.copyTo(copy);
    cv::Scalar color = t == BLACK ? cv::Scalar(255,255,255) :
                       (t == WHITE ? cv::Scalar(0,0,0) : cv::Scalar(0,0,255));
    int baseLine = 0;
    cv::Size textSize = cv::getTextSize(text, cv::QT_FONT_NORMAL, 2, 3, &baseLine);
    int p = pos == TOP ? 100 : image.rows - 100;
    cv::Point bot_left((image.cols - textSize.width)/2, p);
    cv::putText(copy, text, bot_left, cv::QT_FONT_NORMAL, 2, color, 3);
    cv::imshow(window, copy);
    cv::waitKey(HOLD);
}

void addGuide(const cv::Mat& image, cv::Mat& guide) {
    cv::Mat copy;
    image.copyTo(copy);
    cv::Size size(image.cols/3, image.rows/3);
    cv::resize(guide, guide, size);
    for (int r = image.rows - size.height; r < image.rows; r++) {
        for (int c = image.cols - size.width; c < image.cols; c++) {
            cv::Vec3b& des_pixel = copy.at<cv::Vec3b>(r,c);
            cv::Vec3b& src_pixel = guide.at<cv::Vec3b>(r - image.rows + size.height,
                                                       c - image.cols + size.width);
            des_pixel[0] = src_pixel[0];
            des_pixel[1] = src_pixel[1];
            des_pixel[2] = src_pixel[2];
        }
    }
    cv::imshow(window, copy);
    cv::waitKey(IMEDIATE);
}


#endif /* gui_h */
