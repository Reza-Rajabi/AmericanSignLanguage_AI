//
//  cost.h
//  AmericanSignLanguage_AI
//
//  Created by Reza Rajabi on 2020-03-11.
//  Copyright © 2020 RR. All rights reserved.
//

#ifndef cost_h
#define cost_h

#include <opencv2/opencv.hpp>
                                                        // each layer length (L layers)
const int IN_SIZE = 784;        /// same as NUM_FEATURE                 n
const int HIDEN1_SIZE = 28;     ///                                     s1
const int HIDEN2_SIZE = 28;     ///                                     s2
const int HIDEN3_SIZE = 28;     ///                                     s3
const int OUT_SIZE = 24;        /// same as NUM_LABLE                   k

const int NUM_LABLE = 24; /// labeles in (0-25) mapping letter A-Z, but no lable for 9=J or 25=Z because of gesture motions.

const int NUM_LAYER = 5;
const double LAMBDA = 2.0;      /// the regularization factor value to prevent overfitting
const double EPSILON = 0.1;     /// the random initialization of the weights parameter

const int S[NUM_LAYER] = { IN_SIZE, HIDEN1_SIZE, HIDEN2_SIZE, HIDEN3_SIZE, OUT_SIZE }; ///layers size



cv::Mat sigmoid(const cv::Mat& Z) {
    cv::Mat G;
    cv::exp(-Z, G);
    return 1.0 / (1.0 + G);
}

cv::Mat sigmoidPrime(const cv::Mat& Z) {
    cv::Mat G = sigmoid(Z);
    return G.mul(1.0 - G);
}

cv::Mat swish(const cv::Mat& Z) {
    return Z.mul(sigmoid(Z));
}

cv::Mat swishPrime(const cv::Mat& Z) {
    return swish(Z) + sigmoid(Z).mul(1 - swish(Z));
}

struct Activation {
    cv::Mat (*f) (const cv::Mat&) = sigmoid;
    cv::Mat (*fPrime) (const cv::Mat&) = sigmoidPrime;
} AC;

cv::Mat log(const cv::Mat& M) {
    cv::Mat logM;
    cv::log(M, logM);
    return logM;
}

cv::Mat initializeLayerParameters(int inConnections, int outConnections) {
    // NOTE: - randomly initializes parameters for a layer with inConnections number of
    ///        incomming connections and outConnections number of outcomming connections
    ///        it returns a matrice of size (outConnections, 1 + inConnections)
    double epsilon = EPSILON;
    cv::Mat weights = cv::Mat::zeros(outConnections, 1 + inConnections, CV_64F);
    cv::randu(weights, 0.0, 1.0);
    
    return weights * 2 * epsilon - epsilon;
}

void rollTheta(cv::Mat rolled[], const cv::Mat& unrolled) {
    /// the dimention of each Thetha is:  the next layer size x its layer size
    /// Theta0 --> s1 x (n+1)
    /// Theta1 --> s2 x (s1+1)
    /// Theta2 --> s3 x (s2+1)
    /// Theta3 --> k  x (s3+1)
    int rangeStart = 0, rangeEnd = 0;
    cv::Mat transposed[NUM_LAYER-1];
    for (int l = 0; l < NUM_LAYER-1; l++) {
        rangeEnd += S[l+1] * (S[l]+1);
        cv::Range range(rangeStart, rangeEnd);
        cv::Mat temp = unrolled(range, cv::Range::all());
        transposed[l] = cv::Mat(S[l]+1, S[l+1], CV_64F, temp.data); /// transposed column& row
        rangeStart = rangeEnd;
    }
    for (int l = 0; l < NUM_LAYER-1; l++) {
        rolled[l] = transposed[l].t();
    }
}

void unrollTheta(const cv::Mat rolled[], cv::Mat& unrolled) {
    unrolled = cv::Mat::zeros(1, 1, CV_64F);   /// got one extra zero here
    for (int i = 0; i < NUM_LAYER-1; i++) {
        cv::Mat transposed = rolled[i].t();
        cv::Mat temp = cv::Mat(rolled[i].rows * rolled[i].cols ,1, CV_64F, transposed.data);
        cv::vconcat(unrolled, temp, unrolled);
    }
    unrolled = unrolled.rowRange(1, unrolled.rows); /// removed the extra zero here
}

void makeBinaryLables(const cv::Mat& originalLable, cv::Mat& binaryLable) {
    int m = originalLable.rows;
    binaryLable = cv::Mat::zeros(m, OUT_SIZE, CV_64F);
    for (int i = 0; i < m; i++) {
        int columnForOne = originalLable.at<double>(i,0);
        /// we don't have lable 9=J in dataset, so we need to minus one from index 9 to match lables 0 to 24 in 24 cols
        /// and we need to consider to add one later on, when we want to translate a binary row in Y_ to a lable in Y
        if (columnForOne > 9) --columnForOne;
        binaryLable.at<double>(i, columnForOne) = 1;  /// other columns of the row i  is zero
    }
}

void costFunction(Activation A,             /// activation function structure
                  const cv::Mat& params,    /// initial parameters in a rolled up vector
                  const cv::Mat& X,         /// featurs
                  const cv::Mat& Y,         /// lables
                  double lambda,            /// regularization parameter
                  double& J,                /// cost to return
                  cv::Mat& gradient) {      /// gradient to return (partial derivative of cost func.)
    // NOTE: - extracting Theta from vertival vector
    cv::Mat Theta[NUM_LAYER-1];
    rollTheta(Theta, params);
    
    
    // NOTE: - feeding forward and calculating activation parameters
    /// X  --> m x n                                                a0 --> m x (n+1)
    /// a0 * Theta0' --> ( m x (n+1) )  * ( (n+1) x s1 )            a1 --> m x (s1+1)
    /// a1 * Theta1' --> ( m x (s1+1) ) * ( (s1+1) x s2 )           a2 --> m x (s2+1)
    /// a2 * Theta2' --> ( m x (s2+1) ) * ( (s2+1) x s3 )           a3 --> m x (s3+1)
    /// a3 * Theta3' --> ( m x (s3+1) ) * ( (s3+1) x k)             a4 --> m x k
    /// a4 is the hypothesis
    int m = X.rows;
    cv::Mat ones = cv::Mat::ones(m, 1, CV_64F);
    cv::Mat a[NUM_LAYER];
    for(int l = 0; l < NUM_LAYER; l++) {
        a[l] = (l == 0) ? X : A.f(a[l-1] * Theta[l-1].t());
        if(l != NUM_LAYER-1) cv::hconcat(ones, a[l], a[l]);  /// don't add ones to last a[i]
    }
    

    // NOTE: - changing each lable of lables to a vector of (0s and a 1)
    cv::Mat Y_;
    makeBinaryLables(Y, Y_);
    
    
    // NOTE: - calculating cost J
    /// Y_ -->  m x k       a4  --> m x k
    cv::Mat h = a[NUM_LAYER-1];
    J = (-1.0/m) * cv::sum( Y_.mul( log(h) ) + (1 - Y_).mul( log(1 - h) ) )[0];
    
    
    // NOTE: - backpropagation to calcute gradients
    cv::Mat Theta_[NUM_LAYER-1];
    cv::Mat delta[NUM_LAYER-1];
    cv::Mat Theta_g[NUM_LAYER-1];
    /// removing the first column (corresponding to the bias of layer) from each of Theta
    for (int l = 0; l < NUM_LAYER - 1; l++) {
        Theta_[l] = Theta[l].colRange(1, Theta[l].cols);
    }
    /// finding each error using `backpropagation`
    /// delta4 --> delta[3] --> m x k                                            Thetha_g3 --> (k x m)  * (m x s3+1)
    /// delta3 --> delta[2] --> (m x k)  x ( k x s3) . (m x s3)                  Thetha_g2 --> (s3 x m) * (m x s2+1)
    /// delta2 --> delta[1] --> (m x s3) x (s3 x s2) . (m x s2)                  Thetha_g1 --> (s2 x m) * (m x s1+1)
    /// delta1 --> delta[0] --> (m x s2) x (s2 x s1) . (m x s1)                  Thetha_g0 --> (s1 x m) * (m x n+1)
    /// delta0 -->  ------  -->                  ---> we ignore this
    for (int l = NUM_LAYER-2; l >= 0; l--) {
        if (l == NUM_LAYER-2)   /// delta[ i ] : 3->L5, 2->L4, 1->L3, 0->L2, and no error for L1
            delta[l] = (a[l+1] - Y_);
        else                    /// delta[layer l] = delta[l + 1] * Theta[l] . sigmoidPrime(a[l - 1] * Theta[l - 1]
            delta[l] = ( delta[l+1] * Theta_[l+1] ).mul( A.fPrime(a[l] * Theta[l].t()) );
    }
    /// feeding forward again to calculate Theta gredient
    for (int l = 0; l < NUM_LAYER-1; l++) {
        Theta_g[l] = (1.0/m) * delta[l].t() * a[l];
    }

    
    // NOTE: - regularization of the cost and Theta_g to prevent overfitting
    /// we add an extra cv::sum only to convert the result of expression (array of size 1) to a double
    /// needs to set the first column (correspond to the bias of the layer) of Theta to zero (do not regularize bias)
    for (int l = 0; l < NUM_LAYER-1; l++) {
        J += (lambda/(2.0 * m)) * cv::sum( Theta_[l].mul(Theta_[l]) )[0];
        for (int r = 0; r < Theta[l].rows; r++) {
            Theta[l].at<double>(r,0) = 0;
        }
        Theta_g[l] += ((double) lambda/m) * Theta[l];
    }

    
    // NOTE: - roll gradients
    unrollTheta(Theta_g, gradient);
    
    
    // NOTE: Done. J and gradient has been calculated
}


#endif /* cost_h */
