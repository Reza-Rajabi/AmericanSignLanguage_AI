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
#include "cost.h"

const int OPT_ITERATE = 100;        /// maximum iteration for optimizer (train) function
const double OPT_ALPHA = 0.03;      /// the alpha value of the optimizer (train) function
const double OPT_CONVERGE = 1e-9;   /// the min cost amount that would consider as gradient decent has converged


void train(cv::Mat& X, cv::Mat& Y, cv::Mat& Theta, cv::Mat& J_history) {
    J_history = cv::Mat::zeros(OPT_ITERATE, 1, CV_64F);
    cv::Mat Theta_g;
    double J = 0.1; /// something biger than OPT_CONVERGE only to start the loop
    
    std::ofstream ofs;
    ofs.open("jHistory.csv");
    if (ofs.is_open()) ofs << "iter, cost" << std::endl;
    
    for (int i = 0; i < OPT_ITERATE && J > OPT_CONVERGE; i++) {
        if ((i+1) % 10 == 0) std::cout << "Training: " << i+1 << std::endl;
        
        costFunction(Theta, X, Y, lambda, J, Theta_g);
        Theta -= OPT_ALPHA * Theta_g;
        
        J_history.at<double>(i,0) = J;
        if (ofs.is_open()) ofs << i+1 << "," << J << std::endl;
    }
    ofs.close();
}

#endif /* train_h */
