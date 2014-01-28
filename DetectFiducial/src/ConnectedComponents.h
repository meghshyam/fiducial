/*
 * ConnectedComponents.h
 *
 *  Created on: Jan 28, 2014
 *      Author: meghshyam
 */

#ifndef CONNECTEDCOMPONENTS_H_
#define CONNECTEDCOMPONENTS_H_

#include <opencv/cv.h>

int connectedComponents(cv::Mat img, cv::Mat labels, int connectivity, int ltype);


#endif /* CONNECTEDCOMPONENTS_H_ */
