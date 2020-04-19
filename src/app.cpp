//
//  app.cpp
//  AmericanSignLanguage_AI
//
//  Created by Reza Rajabi on 2020-04-16.
//  Copyright © 2020 RR. All rights reserved.
//

#include <opencv2/core.hpp>

#include "predict.h"
#include "preProc.h"
#include "percept.h"


const int INTERVAL = 50;

const char guideFile[] = "asl_guide.png";

// there are no 'J' and 'Z' in the dataset
const char alphabet[] { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                        'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                        'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};


int main() {

    cv::VideoCapture videoStream;
    videoStream.open(0);
    if (!videoStream.isOpened()) {
        std::cout << "Couldn't open camera" << std::endl;
        exit(1);
    }
    
    cv::Mat guide = cv::imread(guideFile, cv::IMREAD_COLOR);
    cv::Mat background;
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    
    // start
    videoStream >> background;
    background *= 0; // black;
    writeOnImage(background, "Talk ASL", COLOR, TOP);
    
    cv::Mat Theta;
    wakeUp(background, Theta);
    
    cv::Mat sample, singleRow;
    int count = 0;
    while (true) {
        if (count == INTERVAL) count = 0;
        
        videoStream >> background;
        
        if (count == INTERVAL - 1) {
            pre_process_image(background, sample, singleRow);
            composeUI(background, sample, guide);
            singleRow = (singleRow - 127.0)/127.0; // normalize
            cv::Mat Predict, Lable;
            predict(AC, singleRow, Theta, Predict);
            lablePredict(Predict, THRESHOLD, Lable);
            int index = Lable.at<double>(0,0);
            if (index != -1) {
                char lable[2] { alphabet[index], '\0' };
                writeOnImage(background, lable, COLOR, BOTT);
            }
            else writeOnImage(background, "Could not undrestand gesture", COLOR, TOP);
        }
        else composeUI(background, sample, guide);

        
        count++;
        cv::waitKey(STREAM);
    }
    
  
    return 0;
}
