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

const int OPT_ITERATE = 600; /// maximum iteration for optimizer function
const double OPT_ALPHA = 0.01;  /// the alpha value of the optimizer function


void train(cv::Mat& X, cv::Mat& Y, cv::Mat& Theta, cv::Mat& J_history) {
    J_history = cv::Mat::zeros(OPT_ITERATE, 1, CV_64F);
    cv::Mat Theta_g = Theta.clone();
    double J = 0.0;
    
    for (int i = 0; i < OPT_ITERATE; i++) {
        costFunction(Theta, X, Y, lambda, J, Theta_g);
        Theta -= OPT_ALPHA * Theta_g;
        J_history.at<double>(i,0) = J;
    }
}

#endif /* train_h */




//
//class Cost: public cv::MinProblemSolver::Function {
//
//    public:
//        void costFunction(const cv::Mat&,const cv::Mat&,
//                          const cv::Mat&,double,double&,cv::Mat&);
//};
//
//void train() {
//    /// creating a solver that uses conjuction gradient optimization algorithm to find the minimum
//    cv::Ptr<cv::ConjGradSolver> fmincg = cv::ConjGradSolver::create();
//    //cv::Ptr<cv::MinProblemSolver::Function> fun(new Cost());
//}
