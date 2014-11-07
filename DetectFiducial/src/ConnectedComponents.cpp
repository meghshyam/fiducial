/*
 * ConnectedComponents.cpp
 *
 *  Created on: Jan 28, 2014
 *      Author: meghshyam
 */

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <vector>
#include "ConnectedComponents.h"

using namespace std;

int connectedComponentsWithStats(const cv::Mat &img, cv::Mat &statsv){
	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;
	cv::Mat cloned_img = img.clone();
	findContours( cloned_img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

	if( !contours.empty() && !hierarchy.empty() )
	{
		// iterate through all the top-level contours,
		vector<cv::Vec4i> cc_box;
		vector<int> cc_area;
		int idx = 0;
		int num_labels = 0;
		for( ; idx >= 0; idx = hierarchy[idx][0] )
		{
			vector<cv::Point> CC = contours[idx];
			cv::Rect rect = boundingRect(CC);
			int left = rect.x;
			int top = rect.y;
			int width = rect.width;
			int height = rect.height;
			int area = contourArea(CC);
			cv::Vec4i sample(left,top, width, height);
			cc_box.push_back(sample);
			cc_area.push_back(area);
			num_labels++;
		}
		statsv.create(num_labels, CC_STAT_MAX, cv::DataType<int>::type);
		for(int i=0; i<num_labels; i++){
			int * row = (int *)& statsv.at<int>(i,0);
			cv::Vec4i sample = cc_box[i];
			row[CC_STAT_LEFT] = sample[0];
			row[CC_STAT_TOP] = sample[1];
			row[CC_STAT_WIDTH] = sample[2];
			row[CC_STAT_HEIGHT] = sample[3];
			row[CC_STAT_AREA] = cc_area[i];
		}
		return num_labels;
	}
	return 0;
}
