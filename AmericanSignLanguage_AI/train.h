//
//  train.h
//  AmericanSignLanguage_AI
//
//  Created by Reza Rajabi on 2020-03-14.
//  Copyright © 2020 RR. All rights reserved.
//

#ifndef train_h
#define train_h

#include <opencv2/opencv.hpp>
#include <chrono>
#include <ctime>
#include "cost.h"

/// counter on terminal
//#define TERMINAL

const int OPT_ITERATE = 100;        /// maximum iteration for optimizer (train) function
const double OPT_ALPHA = 0.03;      /// the alpha value of the optimizer (train) function
const double OPT_CONVERGE = 1e-6;   /// the min cost amount that would consider as gradient descent has converged


void train(Activation ac, cv::Mat& X, cv::Mat& Y, cv::Mat& Theta, cv::Mat& J_history) {
    J_history = cv::Mat::zeros(OPT_ITERATE, 1, CV_64F);
    cv::Mat Theta_g;
    double J = 1.0; /// something biger than OPT_CONVERGE to start the loop
    
    std::ofstream ofs;
    ofs.open("jHistory.csv");
    if (ofs.is_open()) ofs << "iter, cost" << std::endl;
    
    for (int i = 0; i < OPT_ITERATE && J > OPT_CONVERGE; i++) {
    #ifdef TERMINAL
        std::cout << '\r' << "Training: " << i+1 << std::flush;
    #endif
    #ifndef TERMINAL
        if ((i+1)%25 == 0) std::cout << "Training: " << i+1 << std::endl;
    #endif
        
        costFunction(ac, Theta, X, Y, LAMBDA, J, Theta_g);
        Theta -= OPT_ALPHA * Theta_g;
        
        J_history.at<double>(i,0) = J;
        if (ofs.is_open()) ofs << i+1 << "," << J << std::endl;
    }
    ofs.close();
    #ifdef TERMINAL
    std::cout << std::endl;
    #endif
}


#endif /* train_h */
