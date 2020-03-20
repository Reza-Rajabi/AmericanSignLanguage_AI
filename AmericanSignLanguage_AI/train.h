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


void train_gradDescent(cv::Mat& X, cv::Mat& Y, cv::Mat& Theta, cv::Mat& J_history) {
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
        
        costFunction(Theta, X, Y, lambda, J, Theta_g);
        Theta -= OPT_ALPHA * Theta_g;
        
        J_history.at<double>(i,0) = J;
        if (ofs.is_open()) ofs << i+1 << "," << J << std::endl;
    }
    ofs.close();
    #ifdef TERMINAL
    std::cout << std::endl;
    #endif
}


/*------------------------------- THIS IS NOT WORKING --------------------------------*/
// Conjunction Gradient optimisation method
class Cost: public cv::ConjGradSolver::Function {
    cv::Mat _params;     /// initial parameters in a rolled up vector
    cv::Mat _X;          /// featurs
    cv::Mat _Y;          /// lables
    double _lambda;      /// regularization parameter
    double _J;           /// cost to return
    cv::Mat _gradient;   /// gradient to return (partial derivative of cost func.)
    
public:
    Cost(cv::Mat& params, cv::Mat& X, cv::Mat& Y, double lambda, double& J, cv::Mat& gradient) {
        _params = params;
        _X = X;
        _Y = Y;
        _lambda = lambda;
        _J = J;
        _gradient = gradient;
    }
    
    // `Function` abstract class methods
    double calc(const double* j) const {
        cv::Mat params = cv::Mat(getDims(), 1, CV_64F, (void*)j);
        cv::Mat X = _X.clone();
        cv::Mat Y = _Y.clone();
        double lambda = _lambda;
        double J = _J;
        cv::Mat gradient;
        costFunction(params, X, Y, lambda, J, gradient);
        return J;
    }
    int getDims() const {
        return _params.rows;
    }
};

void train_conjGrad(cv::Mat& X, cv::Mat& Y, cv::Mat& Theta) {
    cv::Mat Theta_g;
    double J = 0;

    /// creating a solver that uses conjuction gradient optimization algorithm to find the minimum
    cv::Ptr<cv::MinProblemSolver::Function> f(new Cost(Theta, X, Y, lambda, J, Theta_g));
    cv::Ptr<cv::ConjGradSolver> fmincg = cv::ConjGradSolver::create();
    fmincg->setFunction(f);
    cv::TermCriteria term(cv::TermCriteria::MAX_ITER +
                          cv::TermCriteria::EPS,
                          OPT_ITERATE, OPT_CONVERGE);
    fmincg->setTermCriteria(term);
    std::cout << "Training Start" << std::endl;
    auto start = std::chrono::system_clock::now();
    J = fmincg->minimize(Theta);
    std::cout << "Training Done" << std::endl;
    auto end = std::chrono::system_clock::now();
    std::cout << J << std::endl;
    std::cout << Theta << std::endl;
    
    std::ofstream ofs;
    ofs.open("perception.csv");
    if (ofs.is_open()) {
        for (int r = 0; r < Theta.rows; r++) {
            ofs << Theta.at<double>(r,0) << std::endl;
        }
    }
    ofs.close();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Training Time: " << elapsed_seconds.count() / 60 << " min" << std::endl;
}

#endif /* train_h */
