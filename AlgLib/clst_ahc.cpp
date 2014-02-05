#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "dataanalysis.h"
#include "ap.h"
using namespace alglib;

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

int main(int argc, char **argv)
{
    //
    // The very simple clusterization example
    //
    // We have a set of points in 2D space:
    //     (P0,P1,P2,P3,P4) = ((1,1),(1,2),(4,1),(2,3),(4,1.5))
    //
    //  |
    //  |     P3
    //  |
    //  | P1          
    //  |             P4
    //  | P0          P2
    //  |-------------------------
    //
    // We want to perform Agglomerative Hierarchic Clusterization (AHC),
    // using complete linkage (default algorithm) and Euclidean distance
    // (default metric).
    //
    // In order to do that, we:
    // * create clusterizer with clusterizercreate()
    // * set points XY and metric (2=Euclidean) with clusterizersetpoints()
    // * run AHC algorithm with clusterizerrunahc
    //
    // You may see that clusterization itself is a minor part of the example,
    // most of which is dominated by comments :)
    //
    clusterizerstate s;
    ahcreport rep;
    real_2d_array xy = "[[1,1],[1,2],[4,1],[2,3],[4,1.5]]";

    clusterizercreate(s);
    clusterizersetpoints(s, xy, 2);
    clusterizerrunahc(s, rep);
	integer_2d_array Z(rep.z);
	real_1d_array dist(rep.mergedist);
	int numRows = Z.rows();
	int numCols = Z.cols();
	int numObs = 5;
	for(int i=0; i<numRows; i++)
	{
		int a = Z(i,0);
		int b = Z(i,1);
		printf("%d\t%d\t%.4f\n", a,b,dist[i]);
	}

	int *clusterid = new int[numRows];
	int current_cluster_id = 0;
	//float criteria = atof(argv[1]);
	float criteria = 2.5;
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
				if (a >= numObs){
					a-= numObs;
					done[a] = 1;
					processEntry(a, numObs, current_cluster_id, Z, clusterid, done);
				}else{
					clusterid[a] = current_cluster_id;
				}
			}
			current_cluster_id++;
		}
		i--;
	}

	printf("Cluster INfo\n");
	for(int i=0; i<numObs; i++)
	{
		printf("%d\n", clusterid[i]);
	}

    //
    // Now we've built our clusterization tree. Rep.z contains information which
    // is required to build dendrogram. I-th row of rep.z represents one merge
    // operation, with first cluster to merge having index rep.z[I,0] and second
    // one having index rep.z[I,1]. Merge result has index NPoints+I.
    //
    // Clusters with indexes less than NPoints are single-point initial clusters,
    // while ones with indexes from NPoints to 2*NPoints-2 are multi-point
    // clusters created during merges.
    //
    // In our example, Z=[[2,4], [0,1], [3,6], [5,7]]
    //
    // It means that:
    // * first, we merge C2=(P2) and C4=(P4),    and create C5=(P2,P4)
    // * then, we merge  C2=(P0) and C1=(P1),    and create C6=(P0,P1)
    // * then, we merge  C3=(P3) and C6=(P0,P1), and create C7=(P0,P1,P3)
    // * finally, we merge C5 and C7 and create C8=(P0,P1,P2,P3,P4)
    //
    // Thus, we have following dendrogram:
    //  
    //      ------8-----
    //      |          |
    //      |      ----7----
    //      |      |       |
    //   ---5---   |    ---6---
    //   |     |   |    |     |
    //   P2   P4   P3   P0   P1
    //

//    printf("%s\n", rep.z.tostring().c_str()); // EXPECTED: [[2,4],[0,1],[3,6],[5,7]]

    //
    // We've built dendrogram above by reordering our dataset.
    //
    // Without such reordering it would be impossible to build dendrogram without
    // intersections. Luckily, ahcreport structure contains two additional fields
    // which help to build dendrogram from your data:
    // * rep.p, which contains permutation applied to dataset
    // * rep.pm, which contains another representation of merges 
    //
    // In our example we have:
    // * P=[3,4,0,2,1]
    // * PZ=[[0,0,1,1,0,0],[3,3,4,4,0,0],[2,2,3,4,0,1],[0,1,2,4,1,2]]
    //
    // Permutation array P tells us that P0 should be moved to position 3,
    // P1 moved to position 4, P2 moved to position 0 and so on:
    //
    //   (P0 P1 P2 P3 P4) => (P2 P4 P3 P0 P1)
    //
    // Merges array PZ tells us how to perform merges on the sorted dataset.
    // One row of PZ corresponds to one merge operations, with first pair of
    // elements denoting first of the clusters to merge (start index, end
    // index) and next pair of elements denoting second of the clusters to
    // merge. Clusters being merged are always adjacent, with first one on
    // the left and second one on the right.
    //
    // For example, first row of PZ tells us that clusters [0,0] and [1,1] are
    // merged (single-point clusters, with first one containing P2 and second
    // one containing P4). Third row of PZ tells us that we merge one single-
    // point cluster [2,2] with one two-point cluster [3,4].
    //
    // There are two more elements in each row of PZ. These are the helper
    // elements, which denote HEIGHT (not size) of left and right subdendrograms.
    // For example, according to PZ, first two merges are performed on clusterization
    // trees of height 0, while next two merges are performed on 0-1 and 1-2
    // pairs of trees correspondingly.
    //
  //  printf("%s\n", rep.p.tostring().c_str()); // EXPECTED: [3,4,0,2,1]
  //  printf("%s\n", rep.pm.tostring().c_str()); // EXPECTED: [[0,0,1,1,0,0],[3,3,4,4,0,0],[2,2,3,4,0,1],[0,1,2,4,1,2]]
    return 0;
}
