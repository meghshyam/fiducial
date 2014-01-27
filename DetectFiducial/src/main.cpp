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
#include <iostream>

using namespace cv;
using namespace std;

std::vector<Mat>& findConnectedComponents(Mat &);
int main(int argc, char *argv[])
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
	Mat gaborOutput;
	gaborOutput = gbFilter.filter(input);

	//Find Connected components in Gabor output
	//std::vector<Mat> connectedComponents = findConnectedComponents(gaborOutput);
}

std::vector<Mat> & findConnectedComponents(Mat &binaryImage){
	std::vector<Mat> out;
	return out;
}
