/*
 * GaborFilter.cpp
 *
 *  Created on: Jan 23, 2014
 *      Author: meghshyam
 */

#include "GaborFilter.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/highgui.h>
#include <math.h>
#include <iostream>
using namespace std;
GaborFilter::GaborFilter() {
	// TODO Auto-generated constructor stub

}

GaborFilter::~GaborFilter() {
	// TODO Auto-generated destructor stub
}

void GaborFilter::filter(Mat &input, Mat &gaborOutput){
	Size ksize(64,64);
	double sigma, theta, lambd, gamma, psi1, psi2;
	int ktype = CV_64F;
	int bw =1;
	lambd  = 8; theta   = 0;
	psi1     = 0; psi2 =  M_PI/2;
	gamma   = 0.5;
	sigma = lambd/M_PI*sqrt(log(2)/2)*(pow(2,bw)+1)/(pow(2,bw)-1);
	Mat final_output;
	Mat inputGray, output[8];
	if(input.channels() >=3 )
		cvtColor(input, inputGray, CV_BGR2GRAY);
	else
		inputGray = input.clone();
	//cout<<"Type of imageo:"<<inputGray.type()<<"\n";

	for(int i=0; i<8; i++)
	{
		Mat output1, output2, magOutput;
		Mat kernel1 = getGaborKernel(ksize,sigma,theta,lambd,gamma,psi1);
		filter2D(inputGray, output1, CV_32F, kernel1);
		Mat kernel2 = getGaborKernel(ksize,sigma,theta,lambd,gamma,psi2);
		filter2D(inputGray, output2, CV_32F, kernel2);
		magnitude(output1, output2, magOutput);
		pow(magOutput, 2, output[i]);
		if(i > 0)
		{
			add(final_output, output[i], final_output);
		}else{
			final_output = output[i];
		}
		theta += 45*M_PI/180;
	}
	pow(final_output, 0.5, final_output);
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;

	minMaxLoc( final_output, &minVal, &maxVal, &minLoc, &maxLoc );
	final_output = final_output * (1.0/maxVal);
	gaborOutput = final_output > 0.4;
}
