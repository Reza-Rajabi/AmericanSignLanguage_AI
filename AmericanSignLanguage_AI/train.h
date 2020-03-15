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

class Cost: public cv::MinProblemSolver::Function {

    public:
        void costFunction(const cv::Mat&,const cv::Mat&,
                          const cv::Mat&,double,double&,cv::Mat&);
};

void train() {
    /// creating a solver that uses conjuction gradient optimization algorithm to find the minimum
    cv::Ptr<cv::ConjGradSolver> fmincg = cv::ConjGradSolver::create();
    //cv::Ptr<cv::MinProblemSolver::Function> fun(new Cost());
}

#endif /* train_h */
