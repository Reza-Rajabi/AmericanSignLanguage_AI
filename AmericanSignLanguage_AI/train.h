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
const int BATCH_SIZE = 10000;


void batch(const cv::Mat& depot, cv::Mat& sub, std::vector<int>& index) {
    int m = depot.rows;
    if (index.size() == 0)
        for (int i = 0; i < m; i++) {
            index.push_back(i);
            cv::randShuffle(index);
        }
    for (int r = 0; r < BATCH_SIZE; r++)
        sub.push_back(depot.row(index[r]));
}

void train(Activation ac, const cv::Mat& X, const cv::Mat& Y, cv::Mat& Theta, cv::Mat& J_history) {
    J_history = cv::Mat::zeros(OPT_ITERATE, 1, CV_64F);
    cv::Mat Theta_g;
    double J = 1.0; /// something biger than OPT_CONVERGE to start the loop
    cv::Mat adam = cv::Mat::zeros(Theta.rows, Theta.cols, CV_64F);
    
    std::ofstream ofs;
    ofs.open("jHistory.csv"); //, std::ofstream::app
    if (ofs.is_open()) ofs << "iter, cost" << std::endl;
    
    for (int i = 0; i < OPT_ITERATE && J > OPT_CONVERGE; i++) {
    #ifdef TERMINAL
        std::cout << '\r' << "Training: " << i+1 << std::flush;
    #endif
    #ifndef TERMINAL
        if ((i+1)%25 == 0) std::cout << "Training: " << i+1 << std::endl;
    #endif
        
        cv::Mat batch_X, batch_Y;
        std::vector<int> index;
        batch(X, batch_X, index);
        batch(Y, batch_Y, index);
        
        costFunction(ac, Theta, batch_X, batch_Y, LAMBDA, J, Theta_g);
        adam = 0.999 * adam +  (1 - 0.999) * Theta_g.mul( Theta_g ) + 0.0001;
        cv::sqrt(abs(adam), adam);
        cv::divide(Theta_g, adam, adam);
        Theta = Theta - OPT_ALPHA * adam;
        
        J_history.at<double>(i,0) = J;
        if (ofs.is_open()) ofs << i+1 << "," << J << std::endl;
    }
    ofs.close();
    #ifdef TERMINAL
    std::cout << std::endl;
    #endif
}


#endif /* train_h */
