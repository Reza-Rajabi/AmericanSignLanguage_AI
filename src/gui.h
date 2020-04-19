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

const char window[] = "ASL.talk";


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

void composeUI(cv::Mat& base, cv::Mat machineView, cv::Mat guide) {
    // attach machineView to top left
    if (!machineView.empty()) {
        cv::Mat copy;
        machineView.copyTo(copy);
        cv::Size sizeTop(base.cols/9, base.rows/9);
        cv::resize(copy, copy, sizeTop);
        cv::Mat copy3chanel(machineView.rows, machineView.cols, CV_8UC3);
        cv::cvtColor(copy, copy3chanel, cv::COLOR_GRAY2BGR);
        for (int r = 0; r < sizeTop.height; r++) {
            for (int c = 0; c < sizeTop.width; c++) {
                cv::Vec3b& des_pixel = base.at<cv::Vec3b>(r,c);
                cv::Vec3b src_pixel = copy3chanel.at<cv::Vec3b>(r,c);
                des_pixel[0] = src_pixel[0];
                des_pixel[1] = src_pixel[1];
                des_pixel[2] = src_pixel[2];
            }
        }
    }
    
    // attach the guide to bottom right
    cv::Mat copy(guide);
    cv::Size sizeBott(base.cols/3, base.rows/3);
    cv::resize(copy, copy, sizeBott);
    int bott_row_start = base.rows - sizeBott.height;
    int bott_col_start = base.cols - sizeBott.width;
    for (int r = bott_row_start; r < base.rows; r++) {
        for (int c = bott_col_start; c < base.cols; c++) {
            cv::Vec3b& des_pixel = base.at<cv::Vec3b>(r,c);
            cv::Vec3b src_pixel = copy.at<cv::Vec3b>(r - bott_row_start, c - bott_col_start);
            des_pixel[0] = src_pixel[0];
            des_pixel[1] = src_pixel[1];
            des_pixel[2] = src_pixel[2];
        }
    }
    
    cv::imshow(window, base);
    cv::waitKey(IMEDIATE);
}


#endif /* gui_h */
