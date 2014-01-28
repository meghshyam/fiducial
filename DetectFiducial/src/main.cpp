/*
 * main.cpp
 *
 *  Created on: Jan 24, 2014
 *      Author: meghshyam
 */
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include "GaborFilter.h"
#include "ConnectedComponents.h"
#include <iostream>

using namespace cv;
using namespace std;

void findConnectedComponents(Mat &, vector<vector<Point> > & );
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
	waitKey();
	//Find Connected components in Gabor output
	vector<vector<Point> > connectedComponents;
	findConnectedComponents(gaborOutput, connectedComponents);

	//Cluster the connected components

	//Merge the clusters

	//Detect code in the bounding box

	//Classify the detected code
}

void findConnectedComponents(Mat &binaryImage, vector< vector<Point> > &out){
//	vector<Vec4i> hierarchy;
//	findContours( binaryImage, out, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
//	Mat dst = Mat::zeros(binaryImage.size(), CV_8UC3);
//
//	if( !out.empty() && !hierarchy.empty() )
//	{
//		// iterate through all the top-level contours,
//		// draw each connected component with its own random color
//		int idx = 0;
//		for( ; idx >= 0; idx = hierarchy[idx][0] )
//		{
//			Scalar color( (rand()&255), (rand()&255), (rand()&255) );
//			drawContours( dst, out, idx, color, CV_FILLED, 8, hierarchy );
//		}
//	}
//
//	imshow( "Connected Components", dst );
	Mat components(binaryImage.size(), CV_8UC1);
	connectedComponents(binaryImage, components, 4, CV_16U);
	imshow( "Connected Components", components);
	waitKey();
}
