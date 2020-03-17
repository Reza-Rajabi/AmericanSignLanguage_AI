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

const int OPT_ITERATE = 1000;    /// maximum iteration for optimizer (train) function
const double OPT_ALPHA = 1.0;  /// the alpha value of the optimizer (train) function


void train(cv::Mat& X, cv::Mat& Y, cv::Mat& Theta, cv::Mat& J_history) {
    J_history = cv::Mat::zeros(OPT_ITERATE, 1, CV_64F);
    cv::Mat Theta_g = Theta.clone();
    double J = 0.0;
    
    std::ofstream ofs;
    ofs.open("jHistory.csv");
    if (ofs.is_open()) ofs << "iter, cost" << std::endl;
    
    for (int i = 0; i < OPT_ITERATE; i++) {
        if (i % 100 == 0) std::cout << "Training: " << i << std::endl;
        
        costFunction(Theta, X, Y, lambda, J, Theta_g);
        Theta -= OPT_ALPHA * Theta_g;
        
        J_history.at<double>(i,0) = J;
        if (ofs.is_open()) ofs << i+1 << "," << J << std::endl;
    }
    ofs.close();
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
