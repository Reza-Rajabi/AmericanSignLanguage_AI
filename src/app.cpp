//
//  app.cpp
//  AmericanSignLanguage_AI
//
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
    writeOnImage(background, "Talk ASL", COLOR, TOP);
    
    cv::Mat Theta = wakeUp(background);
    
    cv::Mat sample;
    cv::Mat Predict, Lable;
    int count = 0;
    while (true) {
        if (count == INTERVAL) count = 0;
        videoStream >> background;
        addGuide(background, guide);
        if (count == INTERVAL - 1) {
            sample = pre_process_image(background);
            // doesn't need normalize; we only have one sample
            predict(AC, sample, Theta, Predict);
            lablePredict(Predict, THRESHOLD, Lable);
            int index = Lable.at<double>(0,0);
            if (index != -1) {
                char lable[2] { alphabet[index], '\0' };
                writeOnImage(background, lable, COLOR, BOTT);
            }
            else writeOnImage(background, "Could not undrestand the sign", COLOR, TOP);
        }
        
        count++;
        cv::waitKey(STREAM);
    }
    
  
    return 0;
}
