#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#define FILTER_ONLY /// otherwise applies key-point matching processing too, and cuts the detected hand

using namespace std;
using namespace cv;
using namespace cv::dnn;



string protoFile = "pose_deploy.prototxt";
string weightsFile = "pose_iter_102000.caffemodel";

int nPoints = 22;

int lowHue = 50, lowSat = 0, lowV = 0;
int highHue = 255, highSat = 255, highLum = 255;


void captureAndProcess() {
	cv::namedWindow("Capture", cv::WINDOW_AUTOSIZE);
	//cv::namedWindow("Detected Frame", cv::WINDOW_AUTOSIZE);

	cv::VideoCapture cap;

	cap.open(0);
	if (!cap.isOpened()) {
		cerr << "Couldn't open capture." << std::endl;
	}

	cv::Mat frame, frameHLS, frameDetected, frameFinal;

    char keystroke = 0;
	while (keystroke != 120) {
		cap >> frame;
		cv::imshow("Capture", frame);
		frame.copyTo(frameFinal);

		cv::cvtColor(frame, frameHLS, COLOR_BGR2HLS);
		//cv::GaussianBlur(frameHLS, frameHLS, cv::Size(3, 3), 0);
		frameHLS.copyTo(frameDetected);

		for (int i = 0; i < frame.rows; i++) {
			for (int j = 0; j < frame.cols; j++) {
				cv::Vec3b k = frameHLS.at<cv::Vec3b>(i, j);
				if (k[0] <= 20 || k[0] >= 160 && k[2] >= 20 && (200 < (k[1] / k[2]) && 765 > (k[1] / k[2]))) {}
				else {
					frameFinal.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
				}
			}
		}

		cv::erode(frameFinal, frameFinal, cv::Mat());

		cv::imshow("Capture", frameFinal);

		keystroke = cv::waitKey(33);
		if (keystroke == 120) {
			//taken from https://github.com/spmallick/learnopencv
			//freeze frame for 1 sec. 
			cv::waitKey(1000);

			Mat image;

			frameFinal.copyTo(image);
			
			float thresh = 0.01;

			Mat frameCopy = image.clone();
			Mat cutframe = image.clone();
			Mat reframe;
			int frameWidth = image.cols;
			int frameHeight = image.rows;

			float aspect_ratio = frameWidth / (float)frameHeight;
			int inHeight = 368;
			int inWidth = (int(aspect_ratio*inHeight) * 8) / 8;

			//cout << "inWidth = " << inWidth << " ; inHeight = " << inHeight << endl;

			double t = (double)cv::getTickCount();
			Net net = readNetFromCaffe(protoFile, weightsFile);

			Mat inpBlob = blobFromImage(image, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);

			net.setInput(inpBlob);

			Mat output = net.forward();

			int H = output.size[2];
			int W = output.size[3];

			// find the position of the body parts
			vector<Point> points(nPoints);

			int minx = frame.cols, miny = frame.rows, maxx = 0, maxy = 0;
			for (int n = 0; n < nPoints; n++)
			{
				// Probability map of corresponding body's part.
				Mat probMap(H, W, CV_32F, output.ptr(0, n));
				resize(probMap, probMap, Size(frameWidth, frameHeight));

				Point maxLoc;
				double prob;
				minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
				if (prob > thresh)
				{
					circle(frameCopy, cv::Point((int)maxLoc.x, (int)maxLoc.y), 8, Scalar(0, 255, 255), -1);
					cv::putText(frameCopy, cv::format("%d", n), cv::Point((int)maxLoc.x, (int)maxLoc.y), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);

				}
				points[n] = maxLoc;

				//store min and max values;
				if (points[n].x < minx) minx = points[n].x;

				if (points[n].x > maxx) maxx = points[n].x;

				if (points[n].y < miny) miny = points[n].y;

				if (points[n].y > maxy) maxy = points[n].y;
			}

			// add buffer to min and max to capture the hand 
			Point min = Point(minx - 50, miny - 50);
			Point max = Point(maxx + 50, maxy + 50);

			//if buffered min, max are out of range, keep it within image size. 
			if (max.x > frame.cols) max.x = frame.cols;
			if (max.y > frame.rows) max.y = frame.rows;
			if (min.x < 0) min.x = 0;
			if (min.y < 0) min.y = 0;

			//draw a rectangle of detected hand
			rectangle(frameCopy, min, max, (0, 255, 255));

			cv::Range rows(min.y, max.y);
			cv::Range cols(min.x, max.x);

			//crop detected hand
			cutframe = frame(rows, cols);

			//convert to grayscale and resize to 28 x 28 
			cvtColor(cutframe, cutframe, COLOR_BGR2GRAY);
			Size size(28, 28);
#ifdef FILTER_ONLY
            resize(frame, reframe, size);
#else
			resize(cutframe, reframe, size);
#endif

			//cout << "Min x: " << minx << endl
			//	<< "Min y: " << miny << endl
			//	<< "Max x: " << maxx << endl
			//	<< "Max y: " << maxy << endl;

			t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
			//cout << "Time Taken = " << t << endl;

			imshow("Capture", frameCopy);
			//imshow("Pre-processed", reframe);
			imwrite("preprocessed.jpg", reframe);

		};

		cv::waitKey(100);

	}

}
