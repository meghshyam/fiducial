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
GaborFilter::GaborFilter() {
	// TODO Auto-generated constructor stub

}

GaborFilter::~GaborFilter() {
	// TODO Auto-generated destructor stub
}

Mat& GaborFilter::filter(Mat &input){
	Size ksize(16,16);
	double sigma, theta, lambd, gamma, psi1, psi2;
	int ktype = CV_64F;
	sigma = 4.49;
	lambd  = 8; theta   = 0;
	psi1     = 0; psi2 =  M_PI/2;
	gamma   = 0.5;
	Mat output[8];
	Mat final_output;
	imshow("input", input);
	for(int i=0; i<8; i++)
	{
		Mat kernel1 = getGaborKernel(ksize,sigma,theta,lambd,gamma,psi1/2);
		Mat output1, output_color;
		filter2D(input, output1, CV_32F, kernel1);
		theta += 45*M_PI/180;
		pow(output1, 2, output_color);
		cvtColor(output_color, output[i], CV_RGB2GRAY );
		if(i > 0)
		{
			add(final_output, output[i], final_output);
		}else{
			final_output = output[i];
		}
	}
	final_output = final_output * (1.0/8.0);
	pow(final_output, 0.5, final_output);
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;

	minMaxLoc( final_output, &minVal, &maxVal, &minLoc, &maxLoc );
	final_output = final_output * (1.0/maxVal);
	threshold(final_output, final_output, 0.3, 1, THRESH_BINARY);
	imshow("gabor", final_output);
	waitKey();
	return final_output;
}
