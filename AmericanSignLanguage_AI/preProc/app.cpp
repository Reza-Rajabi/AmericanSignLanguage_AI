//
//  app.cpp
//  AmericanSignLanguage_AI
//
//

#include "utility.h"
#include "train.h"
#include "predict.h"
#include "preprocessing.h"

// there are no 'J' and 'Z'
const char alphabet[] { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                        'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                        'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'};


int main() {
    
    // load what AI has learned before
    std::ifstream ifs;
    std::cout << "Loading learned parameters:" << std::endl;
    openStream("learnedParams-256-iter700.csv", ifs);
    int theta_rows = countRows(ifs);
    cv::Mat Theta = cv::Mat(theta_rows, 1, CV_64F);
    openStream("learnedParams-256-iter700.csv", ifs);
    if (ifs.is_open()) {
        std::string num;
        for (int r = 0; r < theta_rows; r++) {
            std::getline(ifs, num);
            Theta.at<double>(r,0) = std::atof(num.c_str());
        }
    }
    std::cout << theta_rows << " learned parameters loaded." << std::endl;
    std::cout << std::endl;
    
    char option = 0;
    while (option != 32) {
        std::cout << "HOLD `x` to capture the sign.\n" <<
                     "Press space-bar to exit.\n"      <<
                     "Press any key to repeat again."  << std::endl;
        
        cv::destroyAllWindows();
        
        // capture an image from camera and make it ready for ASL translation
        captureAndProcess();
        
        // predict the image captured and lable it
        cv::Mat sample = imread("preprocessed.jpg", IMREAD_GRAYSCALE);
        sample.convertTo(sample, CV_64F, 1.0/255.0);
        cv::Mat singleRow = cv::Mat(1, 784, CV_64F, sample.data);
        //normalize(singleRow);   we only have one sample
        cv::Mat Predict, Lable;
        predict(AC, singleRow, Theta, Predict);
        lablePredict(Predict, THRESHOLD, Lable);
        int index = Lable.at<double>(0,0);
        if (index != -1)
            std::cout << "Sign undrestood: " << alphabet[index] << std::endl;
        else std::cout << "Could not undrestand the sign" << std::endl;
        
        option = waitKey(0);
    }
    
    return 0;
}
