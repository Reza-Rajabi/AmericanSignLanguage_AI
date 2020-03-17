//
//  test.h
//  AmericanSignLanguage_AI
//
//  Created by Reza Rajabi on 2020-03-17.
//  Copyright © 2020 RR. All rights reserved.
//

#ifndef test_h
#define test_h

// evaluates the accuracy of the predicts
/// Test: a matrix of m x NUM_LABLE (look at `cost.h`) representing m number of rows, each row is a test lable (answer),
/// and NUM_LABLE columns consist of all 0 but only one 1 which the index of 1 represents the lable number.
/// Predict: a matrix of m x NUM_LABLE representing m rows of predicted values in NUM_LABLE columns having a
/// value between 0 and 1 which then the index of maximum value in a row represents the lable number for that row
/// PRF: an array of three elements consist of Precision, Recall and F_beta parameters, in order, based on confusion matrix
bool evalFun(cv::Mat& Predict, cv::Mat& Test, double beta, double threshold, double* PRF) {
    /*
     * TP -> has a max value in the Predict row at the same index that Test has 1 and
     *       that max value is grater than threshold
     * FP -> has a max value in the Predict row at the diff index that Test has 1 and
     *       that max value is grater than threshold
     * TN -> has a max value in the Predict which is less than threshold while Test
     *       doesn't have any 1 value for that row
     * FN -> has a max value in the Predict which is less than threshold while Test
     *       has a 1 value at some index for that row
     */
    if (threshold < 0 || threshold > 1) {
        std::cout << "Threshold must be in [0,1]" << std::endl;
        return false;
    }
    if (Predict.rows != Test.rows || Predict.cols != Test.cols) {
        std::cout << "Number of tests and predicts doesn't match" << std::endl;
        return false;
    }
    // defining parameters of confusion matrix
    double TP = 0.0, TN = 0.0, FP = 0.0, FN = 0.0;
    
    int indexOf_one_inTestRow;
    int indexOf_max_inPredRow;
    double maxOfPredRow;
    
    for (int r = 0; r < Test.rows; r++) {
        indexOf_one_inTestRow = -1;
        indexOf_max_inPredRow = -1;
        maxOfPredRow = 0;
        for (int c = 0; c < Test.cols; c++) {
            if (Test.at<double>(r,c) == 1.0) indexOf_one_inTestRow = c;
            if (Predict.at<double>(r,c) > maxOfPredRow) {
                maxOfPredRow = Predict.at<double>(r,c);
                indexOf_max_inPredRow = c;
            }
        }
        // one row has been evaluated. setup confusion matrix parameters for that row
        if (maxOfPredRow > threshold && indexOf_max_inPredRow == indexOf_one_inTestRow)
            TP++; /// true predict indicating the class sample belongs to
        else if (maxOfPredRow > threshold && indexOf_max_inPredRow != indexOf_one_inTestRow) FP++; /// false predict indicating some class that sample mistakely belongs to
        else if (maxOfPredRow < threshold && indexOf_one_inTestRow == -1)
            TN++; /// true predict indicating that sample doesn,t belong to any group
        else if (maxOfPredRow < threshold && indexOf_one_inTestRow > -1)
            FN++; /// false predict indicating that sample doesn,t belong to any group while it does
        else {
            std::cout << "something is wrong at row " << r << std::endl;
            return false;
        }
        
    }
    
    std::cout << "TP: " << TP << " FP: " << FP << " TN: " << TN << " FN: " << FN << std::endl;
    
    if (TP + FN == 0) {
        std::cout << "There were no positive cases in the input data!" << std::endl;
    }
    else if (TP + FP == 0) {
        std::cout << "None of the cases has been predicted positive!" << std::endl;
    }
    else {
        PRF[0] = TP/(TP + FP); /// Precision
        PRF[1] = TP/(TP + FN); /// Recall
        
        double b = pow(beta,2);
        PRF[2] = (1 + b) * PRF[0] * PRF[1] / (b * PRF[0] + PRF[1]); /// F_beta
    }
    
    return true;
}


#endif /* test_h */
