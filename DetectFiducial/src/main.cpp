/*
 * main.cpp
 *
 *  Created on: Jan 24, 2014
 *      Author: meghshyam
 */
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv/cv.h>
#include "GaborFilter.h"
#include "ConnectedComponents.h"
#include <iostream>

using namespace cv;
using namespace std;

void findConnectedComponents(Mat &, Mat & );
void doCluster(Mat&, Mat&);
int main(int argc, char* argv[])
{
	if(argc < 2){
		cout<<"Usage:detectFiducial <filename>\n";
		exit(-1);
	}

	//Read image

	Mat input;
	input = imread(argv[1], 1);

	//Apply Gabor Filter on input image

	GaborFilter gbFilter;
	Mat gaborOutput(input.size(), CV_8UC1);
	gbFilter.filter(input, gaborOutput);
	imshow("gabour output", gaborOutput);
	//Find Connected components in Gabor output
	Mat connectedComponents, clusters;
	findConnectedComponents(gaborOutput, connectedComponents);

	//Cluster the connected components
	doCluster(connectedComponents, clusters);

	//Merge the clusters

	//Detect code in the bounding box

	//Classify the detected code
}

void findConnectedComponents(Mat &binaryImage, Mat &out){
	Mat stats, centroids;
	Mat components(binaryImage.size(), CV_8UC1);
	//connectedComponents(binaryImage, components, 4, CV_16U);
	int numLabels = connectedComponentsWithStats(binaryImage, components, stats, centroids, 4, CV_16U);
	cout<<"Number of components:"<<numLabels-1<<"\n";
	out.create(numLabels-1, 4, DataType<int>::type);
	for(int i=1; i<numLabels; i++)
	{
		//cout<<"area = "<<stats.at<int>(i, CC_STAT_AREA)<<"\n";
		int area = stats.at<int>(i, CC_STAT_AREA);
		if(area > 25)
		{
			vector<Point> pts;
			int x1 = stats.at<int>(i, CC_STAT_LEFT);
			int y1 = stats.at<int>(i, CC_STAT_TOP);
			int x2 = x1 + stats.at<int>(i, CC_STAT_WIDTH);
			int y2 = y1 + stats.at<int>(i, CC_STAT_HEIGHT);
			Point pt1(x1,y1), pt2(x2,y2);
			int * row = (int *)&out.at<int>(i-1,0);
			row[0] = x1; row[1] = y1; row[2] = x2; row[3] = y2;
			rectangle(binaryImage, pt1, pt2, Scalar(255,0,0) );
		}
	}
	imshow( "Connected Components", binaryImage);
	waitKey();
}

void doCluster(Mat &connectedComponents, Mat& clusters)
{
	cvflann::KMeansIndexParams kmean_params(32, 100);
	int numElements = connectedComponents.rows;
	Mat centers(numElements, 4, CV_32F);
	int true_number_clusters = cv::flann::hierarchicalClustering<int, L2<float> >(connectedComponents, centers, kmean_params );
	// since you get less clusters than you specified we can also truncate our matrix.
	centers = centers.rowRange(cv::Range(0,true_number_clusters));

}
