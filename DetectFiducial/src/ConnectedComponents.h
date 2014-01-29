/*
 * ConnectedComponents.h
 *
 *  Created on: Jan 28, 2014
 *      Author: meghshyam
 */

#ifndef CONNECTEDCOMPONENTS_H_
#define CONNECTEDCOMPONENTS_H_

#include <opencv/cv.h>

int connectedComponents(cv::Mat &img, cv::Mat &labels, int connectivity, int ltype);
int connectedComponentsWithStats(cv::Mat &img, cv::Mat &labels, cv::Mat &statsv, cv::Mat &centroids, int connectivity, int ltype);
//! connected components algorithm output formats

enum { CC_STAT_LEFT   = 0,

       CC_STAT_TOP    = 1,

       CC_STAT_WIDTH  = 2,

       CC_STAT_HEIGHT = 3,

       CC_STAT_AREA   = 4,

       CC_STAT_MAX    = 5

     };


#endif /* CONNECTEDCOMPONENTS_H_ */
