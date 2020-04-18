//
//  percept.h
//  AmericanSignLanguage_AI
//
//  Created by Reza Rajabi on 2020-04-17.
//  Copyright © 2020 RR. All rights reserved.
//

#ifndef percept_h
#define percept_h

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "gui.h"
#include "ML_modules/utility.h"


/// load what it has learned so far
#define LOAD_PERCEPTION
/// keep training
//#define TRAIN


const char learnedParams[] = "learnedParams-256-iter700.csv";
const char trainData[] = "sign_mnist_train.csv";


cv::Mat wakeUp(cv::Mat& background) {
    cv::Mat Theta;
#ifdef LOAD_PERCEPTION
    // load what AI has learned before
    writeOnImage(background, "Loading learned parameters...", BLACK, BOTT);
    std::ifstream ifs;
    openStream(learnedParams, ifs);
    int theta_rows = countRows(ifs);
    Theta = cv::Mat(theta_rows, 1, CV_64F);
    openStream(learnedParams, ifs);
    if (ifs.is_open()) {
        std::string num;
        for (int r = 0; r < theta_rows; r++) {
            std::getline(ifs, num);
            Theta.at<double>(r,0) = std::atof(num.c_str());
        }
    }
    writeOnImage(background, "Learned parameters loaded.", BLACK, BOTT);
#else
    cv::Mat Theta_roll[NUM_LAYER-1];
        for(int i = 0; i < NUM_LAYER-1; i++) {
            Theta_roll[i] = initializeLayerParameters(S[i], S[i+1]);
        }
    unrollTheta(Theta_roll, Theta);
#endif
#ifdef TRAIN
    writeOnImage(background, "Loading training data...", BLACK, BOTT);
    int train_rows;
    cv::Mat train_Y; /// LABLES
    cv::Mat train_X; /// FEATURES
    loadData(trainData, train_rows, train_Y, train_X);
    writeOnImage(background, "Training...", BLACK, BOTT);
    cv::Mat J_history;
    cv::Mat X;
    normalize(train_X);
    train(AC, train_X, train_Y, Theta, J_history);
    /// outputs J_history on a csv file. J_history should decrement consistantly to about zero
#endif
    writeOnImage(background, "Ready", BLACK, BOTT);
    return Theta;
}

#endif /* percept_h */
