#include <iostream>
#include <fstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cvblob.h"
#include <math.h>

# define PI           3.14159265358979323846  /* pi */

using namespace cv;
using namespace cvb;
using namespace std;
static int i = 0, j = 0, c;
String output_file = "O.txt";
String image = "O.png";

Mat HSV_threshold(Mat imgHSV, int LowH, int LowS, int LowV, int HighH, int HighS, int HighV)
{
	Mat imgThresholded;

	inRange(imgHSV, Scalar(LowH, LowS, LowV), Scalar(HighH, HighS, HighV), imgThresholded); //Threshold the image

	//morphological opening (removes small objects from the foreground)
	erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

	//morphological closing (removes small holes from the foreground)
	dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

	return imgThresholded;
}

CvBlobs labelling(Mat img_thresholded, IplImage img_origin)
{
	CvBlobs blobs;
	IplImage img_in = img_thresholded;
	IplImage *img_labeled = cvCreateImage(img_thresholded.size(), IPL_DEPTH_LABEL, 1);
	unsigned int result = cvLabel(&img_in, img_labeled, blobs);
	cvFilterByArea(blobs, 1500, 100000);
	cvRenderBlobs(img_labeled, blobs, &img_in, &img_origin);
	for (CvBlobs::const_iterator it = blobs.begin(); it != blobs.end(); ++it)
	{
		{
			CvContourPolygon* polygon = cvConvertChainCodesToPolygon(&(it->second->contour));
			CvContourPolygon* simplePolygon = cvSimplifyPolygon(polygon, 1);
			cvRenderContourPolygon(simplePolygon, &img_origin, Scalar(255, 255, 255));
			for (CvContoursChainCode::iterator it1 = (it->second->internalContours).begin(); it1 != (it->second->internalContours).end(); ++it1)
			{
				CvContourPolygon* polygon = cvConvertChainCodesToPolygon(*it1);
				CvContourPolygon* simplePolygon = cvSimplifyPolygon(polygon, 1);
				if (cvContourPolygonArea(simplePolygon) > 1000){
					cvRenderContourPolygon(simplePolygon, &img_origin, Scalar(0, 0, 0));
				}
			}
			j++;
		}
		i++;
	}
	return blobs;
}

void getDistanceFunction(CvBlobs blobs, Mat imgOrigin, vector<vector<Point> > contours, double distance[361])
{
	int thickness = 0.5;
	int lineType = CV_AA;
	int counter = 0;
	double theta;
	Point p, p1, p2;

	for (CvBlobs::const_iterator it = blobs.begin(); it != blobs.end(); ++it)
	{
		CvPoint center = cvPoint((int)it->second->centroid.x, (int)it->second->centroid.y);
		printf("Center: (%d,%d)\n", center.x, center.y);
		
		printf("Area: %d\n", it->second->area);
		for (theta = 0; theta < 2 * PI; theta += PI / 180)
		{
			p = Point(it->second->centroid.x + 1000 * sin(theta), it->second->centroid.y + 1000 * cos(theta));
			LineIterator it1(imgOrigin, center, p, 8);
			
			int condition = pointPolygonTest(contours[0], it1.pos(), false);

			for (int i = 0; i < it1.count; i++, it1++)
			{
				if (pointPolygonTest(contours[0], it1.pos(), false) != condition) break;
			}

			distance[counter] = cvDistancePointPoint(center, it1.pos());
			counter++;
		}
	}
}

vector<vector<Point>> draw_contours(Mat image, bool &regionDetected)
{
	Mat image_contours;
	int k;
	RNG rng(12345);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	regionDetected = false;

	findContours(image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point>>contours_poly(contours.size());
	vector<Rect>bounding_rect(contours.size());
	vector<Rect>region_of_interest(contours.size());
	vector<Mat>image_roi(contours.size());

	for (k = 0; k<contours.size(); k++)
	{
		// remove areas smaller than 1500
		if (contourArea(contours[k]) > 1500)
		{
			approxPolyDP(Mat(contours[k]), contours_poly[k], cv::arcLength(cv::Mat(contours[k]), true) * 0.005, true);
			bounding_rect[k] = boundingRect(Mat(contours_poly[k]));
			regionDetected = true;
		}
	}

	// draw the polygonal contours, bounding rectangles 
	image_contours = Mat::zeros(image.size(), CV_8UC3);

	for (k = 0; k<contours.size(); k++)
	{
		if (contourArea(contours[k]) > 1500)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(image_contours, contours_poly, k, color, 1, 8, hierarchy, 0, Point());

			region_of_interest[k] = Rect(bounding_rect[k].tl(), bounding_rect[k].br());

			image_roi[k] = image_contours(region_of_interest[k]);

			imshow("Contours", image_roi[0]);
		}
	}

	return contours_poly;
}

bool normalizeDistanceFunction(double distance[361])
{
	double maxValue = distance[0];
	double minValue = distance[0];
	int minIndex = 0;
	double aux[361];

	for (int i = 0; i <= 360; i++)
	{
		if (distance[i] < minValue)
		{
			minValue = distance[i];
			minIndex = i;
		}
		if (distance[i] > maxValue) maxValue = distance[i];
	}

	if (maxValue / minValue < 1.15)
	{
		printf("It's a Circle\n");
		return true;
	}
	
	for (int i = 0; i <= 360; i++) distance[i] = (distance[i] - minValue) / (maxValue - minValue);
	for (int i = minIndex; i <= 360; i++) aux[i - minIndex] = distance[i];
	for (int i = 0; i < minIndex; i++) aux[361 - minIndex + i] = distance[i];
	for (i = 0; i <= 360; i++) distance[i] = aux[i];

	return false;
}

void readShapes(double shapes[361][9])
{
	string line;
	int counter;

	// 0 - Cross
	counter = 0;
	ifstream myfile0("Cross.txt");
	if (myfile0.is_open())
	{
		while (getline(myfile0, line))
		{
			shapes[counter][0] = stod(line);
			counter++;
		}
		myfile0.close();
	}

	// 1 - Diamond
	counter = 0;
	ifstream myfile1("Diamond.txt");
	if (myfile1.is_open())
	{
		while (getline(myfile1, line))
		{
			shapes[counter][1] = stod(line);
			counter++;
		}
		myfile1.close();
	}

	// 2 - Half circle
	counter = 0;
	ifstream myfile2("Half circle.txt");
	if (myfile2.is_open())
	{
		while (getline(myfile2, line))
		{
			shapes[counter][2] = stod(line);
			counter++;
		}
		myfile2.close();
	}

	// 3 - Hexagon
	counter = 0;
	ifstream myfile3("Hexagon.txt");
	if (myfile3.is_open())
	{
		while (getline(myfile3, line))
		{
			shapes[counter][3] = stod(line);
			counter++;
		}
		myfile3.close();
	}

	// 4 - Rectangle
	counter = 0;
	ifstream myfile4("Rectangle.txt");
	if (myfile4.is_open())
	{
		while (getline(myfile4, line))
		{
			shapes[counter][4] = stod(line);
			counter++;
		}
		myfile4.close();
	}

	// 5 - Square
	counter = 0;
	ifstream myfile5("Square.txt");
	if (myfile5.is_open())
	{
		while (getline(myfile5, line))
		{
			shapes[counter][5] = stod(line);
			counter++;
		}
		myfile5.close();
	}

	// 6 - Star
	counter = 0;
	ifstream myfile6("Star.txt");
	if (myfile6.is_open())
	{
		while (getline(myfile6, line))
		{
			shapes[counter][6] = stod(line);
			counter++;
		}
		myfile6.close();
	}

	// 7 - Trapezoid
	counter = 0;
	ifstream myfile7("Trapezoid.txt");
	if (myfile7.is_open())
	{
		while (getline(myfile7, line))
		{
			shapes[counter][7] = stod(line);
			counter++;
		}
		myfile7.close();
	}

	// 8 - Triangle
	counter = 0;
	ifstream myfile8("Triangle.txt");
	if (myfile8.is_open())
	{
		while (getline(myfile8, line))
		{
			shapes[counter][8] = stod(line);
			counter++;
		}
		myfile8.close();
	}

}

void compareDistanceFunctions(double shapes[361][9], double distance[361])
{
	double error[9];
	int shapeNumber = 0;
	double minError;
	
	for (int j = 0; j <= 8; j++)
	{
		error[j] = 0;
		for (int i = 0; i <= 360; i++)
			error[j] += abs(shapes[i][j] - distance[i]);
	}
	
	minError = error[0];

	for (int j = 0; j <= 8; j++)
	{
		if (error[j] < minError)
		{
			minError = error[j];
			shapeNumber = j;
		}
	}

	switch (shapeNumber)
	{
	case 0: printf("It's a Cross\n");
		break;

	case 1: printf("It's a Diamond\n");
		break;

	case 2: printf("It's a Half circle\n");
		break;

	case 3: printf("It's a Hexagon\n");
		break;

	case 4: printf("It's a Rectangle\n");
		break;

	case 5: printf("It's a Square\n");
		break;

	case 6: printf("It's a Star\n");
		break;

	case 7: printf("It's a Trapezoid\n");
		break;

	case 8: printf("It's a Triangle\n");
		break;
	}
}

int main(int argc, char** argv)
{
	while (true)
	{
		Mat imgOrigin;
		Mat imgHSV;
		Mat img_thresholded;
		vector<vector<Point> > contours;
		double distance[361];
		bool regionDetected, circle;
		double shapes[361][9];


		imgOrigin = imread(image, CV_LOAD_IMAGE_COLOR);   // Read the file
		IplImage img_ori = imgOrigin;
		cvtColor(imgOrigin, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

		imshow("Original Image", imgOrigin);


		if (c < 1)
		{
			readShapes(shapes);

			// Black channel
			img_thresholded = HSV_threshold(imgHSV, 0, 0, 0, 179, 70, 45);
			CvBlobs blobs1 = labelling(img_thresholded, img_ori);
			contours = draw_contours(img_thresholded, regionDetected);
			if (regionDetected)
			{
				printf("BLACK CHANNEL\n");
				getDistanceFunction(blobs1, imgOrigin, contours, distance);
				circle = normalizeDistanceFunction(distance);
			}

			// White channel
			img_thresholded = HSV_threshold(imgHSV, 80, 0, 210, 150, 100, 255);
			CvBlobs blobs2 = labelling(img_thresholded, img_ori);
			contours = draw_contours(img_thresholded, regionDetected);
			if (regionDetected)
			{
				printf("WHITE CHANNEL\n");
				getDistanceFunction(blobs2, imgOrigin, contours, distance);
				circle = normalizeDistanceFunction(distance);
			}

			// Yellow channel
			img_thresholded = HSV_threshold(imgHSV, 25, 50, 160, 55, 210, 255);
			CvBlobs blobs3 = labelling(img_thresholded, img_ori);
			contours = draw_contours(img_thresholded, regionDetected);
			if (regionDetected)
			{
				printf("YELLOW CHANNEL\n");
				getDistanceFunction(blobs3, imgOrigin, contours, distance);
				circle = normalizeDistanceFunction(distance);
			}

			// Red channel
			img_thresholded = HSV_threshold(imgHSV, 0, 110, 170, 179, 200, 240);
			CvBlobs blobs4 = labelling(img_thresholded, img_ori);
			contours = draw_contours(img_thresholded, regionDetected);
			if (regionDetected)
			{
				printf("RED CHANNEL\n");
				getDistanceFunction(blobs4, imgOrigin, contours, distance);
				circle = normalizeDistanceFunction(distance);
			}

			// Blue channel
			img_thresholded = HSV_threshold(imgHSV, 90, 110, 100, 110, 255, 255);
			CvBlobs blobs5 = labelling(img_thresholded, img_ori);
			contours = draw_contours(img_thresholded, regionDetected);
			if (regionDetected)
			{
				printf("BLUE CHANNEL\n");
				getDistanceFunction(blobs5, imgOrigin, contours, distance);
				circle = normalizeDistanceFunction(distance);
			}

			// Orange channel
			img_thresholded = HSV_threshold(imgHSV, 0, 150, 195, 15, 200, 235);
			CvBlobs blobs6 = labelling(img_thresholded, img_ori);
			contours = draw_contours(img_thresholded, regionDetected);
			if (regionDetected)
			{
				printf("ORANGE CHANNEL\n");
				getDistanceFunction(blobs6, imgOrigin, contours, distance);
				circle = normalizeDistanceFunction(distance);
			}

			// Light Green channel
			img_thresholded = HSV_threshold(imgHSV, 70, 150, 110, 90, 255, 190);
			CvBlobs blobs7 = labelling(img_thresholded, img_ori);
			contours = draw_contours(img_thresholded, regionDetected);
			if (regionDetected)
			{
				printf("LIGHT GREEN CHANNEL\n");
				getDistanceFunction(blobs7, imgOrigin, contours, distance);
				circle = normalizeDistanceFunction(distance);
			}

			// Purple channel
			img_thresholded = HSV_threshold(imgHSV, 150, 100, 100, 170, 170, 200);
			CvBlobs blobs8 = labelling(img_thresholded, img_ori);
			contours = draw_contours(img_thresholded, regionDetected);
			if (regionDetected)
			{
				printf("PURPLE CHANNEL\n");
				getDistanceFunction(blobs8, imgOrigin, contours, distance);
				circle = normalizeDistanceFunction(distance);
			}

			if (!circle) compareDistanceFunctions(shapes, distance);

			/*ofstream myfile;
			myfile.open(output_file);
			for (int i = 0; i <= 360; i++) myfile << distance[i] << "\n";
			myfile.close();*/
			
			i = 0;
			Mat mtx(&img_ori);
			imshow("labeled image", imgOrigin);
		}
		c++;

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			//cvReleaseImage(&img_in);
			break;
		}

	}
	return 0;
}