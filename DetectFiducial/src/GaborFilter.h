/*
 * GaborFilter.h
 *
 *  Created on: Jan 23, 2014
 *      Author: meghshyam
 */

#ifndef GABORFILTER_H_
#define GABORFILTER_H_

#include<opencv/cv.h>

using namespace cv;
class GaborFilter {
public:
	GaborFilter();
	virtual ~GaborFilter();
	void filter(Mat &, Mat&);
};

#endif /* GABORFILTER_H_ */
