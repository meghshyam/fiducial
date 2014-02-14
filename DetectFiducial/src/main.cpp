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
#include <opencv/ml.h>
#include <dataanalysis.h>
#include <ap.h>
#include <iostream>

//#define TRAIN_MODE
//#define DEBUG
//#define DEBUG2
//#define DEBUG3

using namespace cv;
using namespace std;
using namespace alglib;

void findConnectedComponents(const Mat &, Mat & );
void doCluster(const Mat&, Mat&);
void getPoints(const Mat &binOutput, int left, int top, int right, int bottom, Mat &outPts);
bool detectCode(const Mat &pts, int left, int top, int right, int bottom, const Mat &input, vector<float>& intensity_profile );
void createTrainingData(const string &orig_dirname, Mat & trainingData, Mat & response);

void processEntry(int index, int numObs, int clusterIndex, integer_2d_array Z, int * clusterid, int *done)
{
	for(int j=0; j<2; j++){
		int a = Z(index,j);
		if (a >= numObs){
			a-= numObs;
			done[a] = 1;
			processEntry(a, numObs, clusterIndex, Z, clusterid, done);
		}else{
			clusterid[a] = clusterIndex;
		}
	}
}

int main(int argc, char* argv[])
{
	if(argc < 2){
		cout<<"Usage:detectFiducial <filename>\n";
		exit(-1);
	}

	//Read image
	Mat input;
	input = imread(argv[1], 1);
	int width = input.cols;
	int height = input.rows;
	//Apply Gabor Filter on input image
	GaborFilter gbFilter;
	Mat gaborOutput(input.size(), CV_8UC1);
	gbFilter.filter(input, gaborOutput);
#ifdef DEBUG
	Mat clonedgaborOutput = gaborOutput.clone();
#endif

#ifndef TRAIN_MODE
	//Find Connected components in Gabor output
	Mat connectedComponents, clusters;
	findConnectedComponents(gaborOutput, connectedComponents);

	//Cluster the connected components
	doCluster(connectedComponents, clusters);
	int numClusters = clusters.rows;
#ifdef DEBUG
	Mat cloned_input = input.clone();
#endif
	for(int i=0; i<numClusters; i++)
	{
		float * row = (float *)&clusters.at<float>(i,0);
		int x1 = row[0];
		int y1 = row[1];
		int x2 = row[2];
		int y2 = row[3];
		Point pt1(x1,y1), pt2(x2,y2);
#ifdef DEBUG
		rectangle(cloned_input, pt1, pt2, Scalar(0,255,0) );
#endif
	}
#ifdef DEBUG
	imshow("box",cloned_input);
	waitKey();
#endif

	//Detect code in the bounding box
	vector<float> intensity_profile;
	bool found = false;
	for(int i=0; i<numClusters; i++)
	{
		float * row = (float *)&clusters.at<float>(i,0);
		int x1 = row[0];
		int y1 = row[1];
		int x2 = row[2];
		int y2 = row[3];
		Mat pts;
		Point pt1(x1,y1), pt2(x2,y2);
#ifdef DEBUG2
	  rectangle(clonedgaborOutput, pt1, pt2, Scalar(255,0,0) );
	  imshow("gabor", clonedgaborOutput);
	  waitKey();
#endif
		getPoints(gaborOutput, x1, y1, x2, y2, pts);
		/*
		x1 = x1 - 5; y1=y1-5; x2 = x2+5; y2 = y2+5;
		x1 = x1 < 0 ? 0 : x1;
		y1 = y1 < 0 ? 0 : y1;
		x2 = x2 > width ? width : x2;
		y2 = y2 > height ? height : y2;
		*/
		found = detectCode(pts, x1, y1, x2, y2, input, intensity_profile);
		if(found)
			break;
	}
	if(!found){
		cout<<"Code not detected\n";
	}else{
		//cout<<"Code detected\n";
		//Classify the detected code
		Mat sample(1, 100, DataType<float>::type);
		Mat profile_mat(intensity_profile);
		transpose(profile_mat, profile_mat);
		resize(profile_mat, sample, sample.size());


		Mat trainingData,response;
		string dirname = "/home/meghshyam/git/fiducial/DetectFiducial/trainingData/";
		createTrainingData(dirname, trainingData, response);

		/*
		FileStorage file("test2.xml", FileStorage::WRITE);
		file<<"Orig_sample"<<profile_mat;
		file<<"Sample"<<sample;
		file.release();
		*/

		const int K =5;
		CvKNearest knn(trainingData, response, Mat(), false, K);
		Mat results(sample.rows, 1, DataType<float>::type);
		knn.find_nearest(sample, K, &results);
		cout<<"Class:"<<results.at<float>(0)<<"\n";
	}
#else
	int x1 = 0;
	int y1 = 0;
	int x2 = input.cols;
	int y2 = input.rows;
	Mat pts;
	vector<float> intensity_profile;
	getPoints(gaborOutput, x1, y1, x2, y2, pts);
	bool found = detectCode(pts, x1, y1, x2, y2, input, intensity_profile);
	if(found){
		Mat sample(1, 100, DataType<float>::type);
		Mat profile_mat(intensity_profile);
		transpose(profile_mat, profile_mat);
		resize(profile_mat, sample, sample.size());

		Mat trainingData,response;
		string dirname = "/home/meghshyam/git/fiducial/DetectFiducial/trainingData/";
		createTrainingData(dirname, trainingData, response);
		/*
			FileStorage file("training.xml", FileStorage::WRITE);
			file<<"Sample"<<trainingData;
			file<<"class"<<response;
		*/
		const int K =5;
		CvKNearest knn(trainingData, response, Mat(), false, K);
		Mat results(sample.rows, 1, DataType<float>::type);
		knn.find_nearest(sample, K, &results);
		cout<<"Class:"<<results.at<float>(0)<<"\n";
	}
#endif

}

bool detectCode(const Mat &pts, int left, int top, int right, int bottom, const Mat &input, vector<float>& intensity_profile ){
	int numPts = pts.rows;
	int dim = pts.cols;
	PCA pca(pts, Mat(), CV_PCA_DATA_AS_ROW, 2);
	float * eigen_vect = pca.eigenvectors.ptr<float>(0);
	double slope = eigen_vect[1]/eigen_vect[0];
	int centroid[2] = {0,0};
	for(int i=0; i<numPts; i++){
		centroid[0] += pts.at<float>(i,0);
		centroid[1] += pts.at<float>(i,1);
	}
	centroid[0] /= numPts;
	centroid[1] /= numPts;

	int centroid_pos[2] = {0, 0};
	int centroid_neg[2] = {0, 0};
	float slope_perp = -1/slope;
	if (abs(slope_perp) <= 1){
		centroid_pos[0] = centroid[0] + 5;
		centroid_neg[0] = centroid[0] - 5;
		centroid_pos[1] = centroid[1] + 5*slope_perp;
		centroid_neg[1] = centroid[1] - 5*slope_perp;
	}else{
		centroid_pos[1] = centroid[1] + 5;
		centroid_neg[1] = centroid[1] - 5;
		centroid_pos[0] = centroid[0] + 5/slope_perp;
		centroid_neg[0] = centroid[0] - 5/slope_perp;
	}

	int centroids[3][2];
	centroids[0][0] = centroid_neg[0]; centroids[0][1] = centroid_neg[1];
	centroids[1][0] = centroid[0]; centroids[1][1] = centroid[1];
	centroids[2][0] = centroid_pos[0]; centroids[2][1] = centroid_pos[1];

	Mat grayInput;//(input.size(), DataType<float>::type);
	//cout<<"Type of image1:"<<grayInput.type()<<"\n";
	if(input.channels() >=3 )
		cvtColor(input, grayInput, CV_RGB2GRAY);
	else
		grayInput = input.clone();
	//cout<<"Type of image1:"<<grayInput.type()<<"\n";

	int numPtsPerLine[3];
	uchar * intensities[3];
	vector<Point> line_pts[3];

	for(int index=0; index<3; index++)
	{
		vector<Point> upward_pts;
		vector<Point> downward_pts;

		bool increase_x = true;
		if (abs(slope) > 1)
			increase_x = false;

		if (increase_x){
			int left_distance = centroid[0] - left;
			int right_distance = right - centroids[index][0];
			if (slope < 0){
				for (int i = 0; i<right_distance; i++){
					int x = centroids[index][0] + i;
					int y = centroids[index][1]  + i*slope;
					if(y >= top)
						upward_pts.push_back(Point(x, y));
					else
						break;
				}
				for (int i = 1; i<left_distance; i++){
					int x = centroids[index][0] - i;
					int y = centroids[index][1] - i*slope;
					if(y <= bottom)
						downward_pts.push_back(Point(x, y));
					else
						break;
				}
			}//slope < 0
			else{
				for (int i = 0; i<left_distance; i++){
					int x = centroids[index][0] - i;
					int y = centroids[index][1] - i*slope;
					if(y >= top)
						upward_pts.push_back(Point(x, y));
					else
						break;
				}
				for (int i = 0; i<right_distance; i++){
					int x = centroids[index][0] + i;
					int y = centroids[index][1] + i*slope;
					if(y <= bottom)
						downward_pts.push_back(Point(x, y));
					else
						break;
				}
			}//slope > 0
		}//} increase_x
		else{
			int up_distance = centroids[index][1]-top;
			int down_distance = bottom - centroids[index][1];
			if (slope < 0){
				for (int i = 0; i< up_distance -1; i++){
					int y = centroids[index][1] - i;
					int x = centroids[index][0] - i/slope;
					if(x <= right)
						upward_pts.push_back(Point(x, y));
					else
						break;
				}
				for (int i = 1; i< down_distance; i++){
					int y = centroids[index][1] + i;
					int x = centroids[index][0] + i/slope;
					if(x >= left)
						downward_pts.push_back(Point(x, y));
					else
						break;
				}
			}
			else{
				for (int i = 0; i< up_distance -1; i++){
					int y = centroids[index][1] - i;
					int x = centroids[index][0] - i/slope;
					if(x >= left)
						upward_pts.push_back(Point(x, y));
					else
						break;

				}
				for (int i = 1; i< down_distance; i++){
					int y = centroids[index][1] + i;
					int x = centroids[index][0] + i/slope;
					if(x <= right)
						downward_pts.push_back(Point(x, y));
					else
						break;

				}
			}
		}

		/*if(upward_pts.size() > 0 && downward_pts.size() > 0){
			Point first_pt = upward_pts.back();
			Point last_pt = downward_pts.back();
			line(input, first_pt, last_pt, Scalar(0,255,0));
		}*/

		reverse(upward_pts.begin(), upward_pts.end());
		line_pts[index] = upward_pts;
		line_pts[index].insert(line_pts[index].end(), downward_pts.begin(), downward_pts.end());
		int totalPoints = upward_pts.size() + downward_pts.size();
		intensities[index] = new uchar[totalPoints];
		numPtsPerLine[index] = totalPoints;
		bool first=true;
		for(int i=0; i<totalPoints; i++){
			Point pt = line_pts[index][i];
			uchar intensity_test = grayInput.at<uchar>(pt);
			intensities[index][i] = intensity_test;
		}
	}
#ifdef DEBUG
	Mat cloned_input = input.clone();
#endif
	vector<int> ring_starts[3];
	vector<int> ring_widths[3];
	for(int i=0; i<3; i++){
		int totalPoints = numPtsPerLine[i];
		Mat profile(totalPoints, 1, CV_8U, intensities[i]);
		//Mat smoothProfile;
		//boxFilter(profile, smoothProfile, CV_32FC1, Size(5,1));
		Mat thresh_profile;
		threshold(profile, thresh_profile, 127, 1, CV_THRESH_BINARY);
		bool start_ring = false;
		totalPoints = thresh_profile.rows;
		uchar * int_profile = thresh_profile.ptr<uchar>(0);
		for(int j=1; j<totalPoints; j++){
			int start,width;
			uchar intensity1 = int_profile[j-1];
			uchar intensity2 = int_profile[j];
			if(  intensity1==0 && intensity2 == 1){
				start_ring = true;
				start = j;
				Point pt = line_pts[i][j];
				uchar intensity_test = profile.at<uchar>(j);
#ifdef DEBUG
				circle(cloned_input, pt,3,Scalar(255,0,0));
#endif
			}
			if (start_ring && intensity2 == 0){
				width = j - start;
				ring_starts[i].push_back(start);
				ring_widths[i].push_back(width);
				start_ring = false;
			}
		}
	}

	int size1 = ring_starts[0].size();
	int size2 = ring_starts[1].size();
	int size3 = ring_starts[2].size();

	//if(size1 == size2 && size2 == size3){
	if(size1 >= 4 && size2 >= 4 && size3 >= 4)
	{
		/*
			int* max_width = new int[size1];
			int *max_index = new int[size1];
			for(int i=0; i<size1; i++){
				int maxwidth = ring_widths[0][i];
				int maxindex = 0;
				for(int j=1; j<3; j++){
					if(ring_widths[j][i] > maxwidth){
						maxindex = j;
						maxwidth = ring_widths[j][i];
					}
				}
				max_width[i] = maxwidth;
				max_index[i] = maxindex;
			}
		 */
		//int first_ring = max_index[0];
		int first_ring = 1;
		int firstRingStartIndex = ring_starts[first_ring][0];
		Point firstRingStartPoint = line_pts[first_ring][firstRingStartIndex];

		//int last_ring = max_index[size1-1];
		int last_ring = 1;
		int lastRingStartIndex = ring_starts[last_ring][size2-1];
		int lastRingWidth = ring_widths[last_ring][size2-1];
		Point lastRingStartPoint = line_pts[last_ring][lastRingStartIndex];
		int xx = lastRingStartPoint.x;
		int yy = lastRingStartPoint.y;
		Point lastRingEndPoint;
		if(slope>0){
			lastRingEndPoint.x = xx + lastRingWidth*abs(eigen_vect[0]);
			lastRingEndPoint.y = yy + lastRingWidth*abs(eigen_vect[1]);
		}
		else{
			lastRingEndPoint.x = xx - lastRingWidth*abs(eigen_vect[0]);
			lastRingEndPoint.y = yy + lastRingWidth*abs(eigen_vect[1]);;
		}
#ifdef DEBUG
		circle(cloned_input, firstRingStartPoint, 3, Scalar(0,0,255) );
		circle(cloned_input, lastRingEndPoint, 3, Scalar(0,0,255) );
#endif
		Point2f diff_vector = lastRingEndPoint -  firstRingStartPoint;
		int len = sqrt(diff_vector.dot(diff_vector));
		diff_vector *= 1.0/len;
		for (int i=0; i<len; i++){
			Point2f diff_vectorWeighted =  i*diff_vector;
			Point sample = firstRingStartPoint + Point(diff_vectorWeighted.x, diff_vectorWeighted.y);
			uchar intensity = grayInput.at<uchar>(sample);
			if(intensity > 127){
				intensity_profile.push_back(1);
			}
			else{
				intensity_profile.push_back(0);
			}
		}
#ifdef DEBUG
		imshow("lines", cloned_input);
		waitKey();
#endif
		return true;
	}
	//}
#ifdef DEBUG
		imshow("lines", cloned_input);
		waitKey();
#endif
	return false;
}

void getPoints(const Mat &binOutput, int left, int top, int right, int bottom, Mat &outPts){
	int rowIndex = 0;
	int maxPts = (right-left+1)*(bottom-top+1);
	outPts.create(maxPts, 2, DataType<float>::type);
	for(int i=top; i<=bottom; i++)
	{
		uchar* row = (uchar*)&binOutput.at<uchar>(i,0);
		for(int j=left; j<=right; j++){
			if(row[j] != 0 ){
				outPts.at<float>(rowIndex,0) = (float)j;
				outPts.at<float>(rowIndex,1) = (float)i;
				rowIndex++;
			}
		}
	}
	outPts = outPts.rowRange(cv::Range(0,rowIndex));
}

void findConnectedComponents(const Mat &binaryImage, Mat &out){
	//Mat clone_image = binaryImage.clone();
	Mat stats;
	int numLabels = connectedComponentsWithStats(binaryImage, stats);
	out.create(numLabels, 4, DataType<float>::type);
	int k=0;
	for(int i=0; i<numLabels; i++)
	{
		//cout<<"area = "<<stats.at<int>(i, CC_STAT_AREA)<<"\n";
		int area = stats.at<int>(i, CC_STAT_AREA);
		if(area > 25)
		{
			int x1 = stats.at<int>(i, CC_STAT_LEFT);
			int y1 = stats.at<int>(i, CC_STAT_TOP);
			int x2 = x1 + stats.at<int>(i, CC_STAT_WIDTH);
			int y2 = y1 + stats.at<int>(i, CC_STAT_HEIGHT);
			float * row = (float*)&out.at<float>(k,0);
			k++;
			row[0] = (float)x1; row[1] = (float)y1; row[2] = (float)x2; row[3] = (float)y2;
			Point pt1(x1,y1), pt2(x2,y2);
			/*
			rectangle(clone_image, pt1, pt2, Scalar(255,0,0) );
			imshow("connected comps", clone_image);
			waitKey();
			*/
		}
	}

	if(k<numLabels){
		out = out.rowRange(cv::Range(0,k));
	}
	stats.release();
}

void doCluster(const Mat &connectedComponents, Mat& clusters)
{
	int numElements = connectedComponents.rows;
	if(numElements == 1){
		clusters.create(1,4,DataType<float>::type);
		float * row = (float *)&connectedComponents.at<float>(0,0);
		float *cluster= (float*) &clusters.at<float>(0,0);
		cluster[0]= row[0];
		cluster[1]= row[1];
		cluster[2]= row[2];
		cluster[3]= row[3];
		return;
	}

	/*cvflann::KMeansIndexParams kmean_params(10, 30, cvflann::CENTERS_KMEANSPP);
	Mat centers(numElements, 4, CV_32F);
	int true_number_clusters = cv::flann::hierarchicalClustering<cvflann::L2<float> >(connectedComponents, centers, kmean_params);
	// since you get less clusters than you specified we can also truncate our matrix.
	centers = centers.rowRange(cv::Range(0,true_number_clusters));
	clusters.create(true_number_clusters, 4, DataType<float>::type);
	 */

	real_2d_array input;
	double* pContent = new double[numElements*4];
	for (int i=0; i<numElements; i++){
		float * row = (float *)&connectedComponents.at<float>(i,0);
		for(int j =0; j<4; j++){
			int index = i*4 + j;
			pContent[index] = (double)row[j];
		}
	}

	input.setcontent(numElements, 4, pContent);
	int *clusterid = new int[numElements];

	clusterizerstate s;
	ahcreport rep;

	clusterizercreate(s);
	clusterizersetahcalgo(s,2);
	clusterizersetpoints(s, input, 2);
	clusterizerrunahc(s, rep);
	integer_2d_array Z(rep.z);
	real_1d_array dist(rep.mergedist);
	int numRows = Z.rows();

	int current_cluster_id = 0;
	//float criteria = atof(argv[1]);
	float criteria = 150;
	int *done = new int[numRows];
	for(int i=0; i<numRows; i++)
	{
		done[i] = 0;
	}

	int i=numRows-1;
	while(i >= 0){
		if(!done[i] && dist(i) <= criteria){
			for(int j=0; j<2; j++){
				int a = Z(i,j);
				if (a >= numElements){
					a-= numElements;
					done[a] = 1;
					processEntry(a, numElements, current_cluster_id, Z, clusterid, done);
				}else{
					clusterid[a] = current_cluster_id;
				}
			}
			current_cluster_id++;
		}else if(!done[i]){
			for(int j=0; j<2; j++){
				int a = Z(i,j);
				if (a < numElements){
					clusterid[a] = current_cluster_id;
					current_cluster_id++;
				}
			}
		}
		i--;
	}

	int numClusters = current_cluster_id;
	clusters.create(numClusters,4,DataType<float>::type);
	int * firstMember = new int[numClusters];
	for(int i=0; i<numClusters; i++)
	{
		firstMember[i] = 0;
	}
	/*
	printf("Cluster INfo\n");
	for(int i=0; i<numElements; i++)
	{
		printf("%d\n", clusterid[i]);
	}
	 */

	for(int i=0; i<numElements; i++)
	{
		int minIndex = clusterid[i];
		float * row = (float *)&connectedComponents.at<float>(i,0);
		float *cluster= (float*) &clusters.at<float>(minIndex,0);
		if(firstMember[minIndex] == 0)
		{
			firstMember[minIndex] = 1;
			cluster[0]= row[0];
			cluster[1]= row[1];
			cluster[2]= row[2];
			cluster[3]= row[3];
		}
		else{
			if(cluster[0] > row[0])
				cluster[0] = row[0];
			if(cluster[1] > row[1])
				cluster[1] = row[1];
			if(cluster[2] < row[2])
				cluster[2] = row[2];
			if(cluster[3] < row[3])
				cluster[3] = row[3];
		}
	}
	delete(firstMember);
	delete(clusterid);
}

void createTrainingData(const string &orig_dirname, Mat & trainingData, Mat & response)
{
	string dirname = orig_dirname;
	string filename = dirname.append("trainingdata00.yaml");
	FileStorage file1(filename, FileStorage::READ);
	Mat trainingData00;
	file1["training_data_00"] >> trainingData00;
	file1.release();

	dirname = orig_dirname;
	filename = dirname.append("trainingdata01.yaml");
	FileStorage file2(filename, FileStorage::READ);
	Mat trainingData01;
	file2["training_data_01"] >> trainingData01;
	file2.release();

	dirname = orig_dirname;
	filename = dirname.append("trainingdata10.yaml");
	FileStorage file3(filename, FileStorage::READ);
	Mat trainingData10;
	file3["training_data_10"] >> trainingData10;
	file3.release();

	dirname = orig_dirname;
	filename = dirname.append("trainingdata11.yaml");
	FileStorage file4(filename, FileStorage::READ);
	Mat trainingData11;
	file4["training_data_11"] >> trainingData11;
	file4.release();

	vconcat(trainingData00, trainingData01, trainingData);
	vconcat(trainingData, trainingData10, trainingData);
	vconcat(trainingData, trainingData11, trainingData);

	Mat response1(trainingData00.rows, 1, DataType<float>::type);
	Mat response2(trainingData01.rows, 1, DataType<float>::type);
	Mat response3(trainingData10.rows, 1, DataType<float>::type);
	Mat response4(trainingData11.rows, 1, DataType<float>::type);
	response1 = Scalar(0); response2 =Scalar(1); response3 =Scalar(2); response4 =Scalar(3);

	vconcat(response1, response2, response);
	vconcat(response, response3, response);
	vconcat(response, response4, response);
}
