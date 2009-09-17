#include <iostream>
#include "cvcgsolve.h"

void icvMatMulOps( CvMat* X, CvMat* Y, void* userdata )
{
	CvMat* A = (CvMat*)userdata;
	cvMatMul( A, X, Y );
}

int main()
{
	CvMat* A = cvCreateMat( 3, 3, CV_64FC1 );
	CvMat* B = cvCreateMat( 3, 1, CV_64FC1 );
	CvMat* X = cvCreateMat( 3, 1, CV_64FC1 );
	A->data.db[0] = 11;
	A->data.db[1] = 5;
	A->data.db[2] = 15;
	A->data.db[3] = 5;
	A->data.db[4] = 3;
	A->data.db[5] = 1;
	A->data.db[6] = 15;
	A->data.db[7] = 1;
	A->data.db[8] = 31;
	B->data.db[0] = 1;
	B->data.db[1] = 2;
	B->data.db[2] = 3;
	printf("A :\n[%lf, %lf, %lf]\n[%lf, %lf, %lf]\n[%lf, %lf, %lf]\n", A->data.db[0], A->data.db[1], A->data.db[2], A->data.db[3], A->data.db[4], A->data.db[5], A->data.db[6], A->data.db[7], A->data.db[8]);
	cvCGSolve( icvMatMulOps, A, B, X );
	printf("x : [%lf, %lf, %lf]\n", X->data.db[0], X->data.db[1], X->data.db[2]);
	cvMatMul( A, X, X );
	printf("Ax : [%lf, %lf, %lf]\n", X->data.db[0], X->data.db[1], X->data.db[2]);
	cvSolve( A, B, X );
	printf("x : [%lf, %lf, %lf]\n", X->data.db[0], X->data.db[1], X->data.db[2]);
	cvMatMul( A, X, X );
	printf("Ax : [%lf, %lf, %lf]\n", X->data.db[0], X->data.db[1], X->data.db[2]);
	return 0;
}
