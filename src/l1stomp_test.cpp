#include "cvl1stomp.h"

void icvAOps( CvMat* X, CvMat* Y, CvMat* I, void* userdata )
{
	CvMat* A = (CvMat*)userdata;
	double* Ap = A->data.db;
	double* Yp = Y->data.db;
	for ( int i = 0; i < A->rows; ++i, ++Yp )
	{
		double y = 0;
		double* Xp = X->data.db;
		int* Ip = I->data.i;
		for ( int j = 0; j < A->cols; ++j, ++Ip )
		{
			if ( *Ip )
			{
				y += (*Xp) * (*Ap);
				++Xp; // NOTE: only move forward when it has value.
			}
			++Ap;
		}
		*Yp = y;
	}
}

void icvAtOps( CvMat* X, CvMat* Y, CvMat* I, void* userdata )
{
	cvZero( Y );
	CvMat* A = (CvMat*)userdata;
	double* Xp = X->data.db;
	double* Ap = A->data.db;
	for ( int i = 0; i < A->rows; ++i, ++Xp )
	{
		double y = 0;
		double* Yp = Y->data.db;
		int* Ip = I->data.i;
		for ( int j = 0; j < A->cols; ++j, ++Ip )
		{
			if ( *Ip )
			{
				*Yp += (*Xp) * (*Ap);
				++Yp; // NOTE: only move forward when it has value.
			}
			++Ap;
		}
	}
}

int main()
{
	const int N = 1024;
	const int K = 128;
	const int T = 20;
	CvMat* A = cvCreateMat( K, N, CV_64FC1 );
	CvMat* X = cvCreateMat( N, 1, CV_64FC1 );
	CvMat* Y = cvCreateMat( K, 1, CV_64FC1 );
	CvRNG rng_state = cvRNG(0xffffffff);
	cvRandArr( &rng_state, A, CV_RAND_NORMAL, cvScalar(0), cvScalar(1) );
	cvZero( X );
	for ( int i = 0; i < T; i++ )
	{
		int idx = cvRandInt( &rng_state ) % N;
		X->data.db[idx] = (int)(cvRandInt( &rng_state ) % 3) - 1;
	}
	//double norm = cvNorm( X );
	//for ( int i = 0; i < N; i++ )
	//	X->data.db[i] = X->data.db[i] / norm;
	double sigma = .005;
	CvMat* e = cvCreateMat( K, 1, CV_64FC1 );
	//cvZero( e );
	cvRandArr( &rng_state, e, CV_RAND_NORMAL, cvScalar(0), cvScalar(sigma) );
	cvMatMulAdd( A, X, e, Y );
	double epsilon = .5;//sigma * sqrt(K) * sqrt(1 + 2 * sqrt(2) / sqrt(K));
	CvMat* X0 = cvCreateMat( N, 1, CV_64FC1 );
	printf("||X0 - X|| Before L1StOMP : %f\n", cvNorm(X0, X, CV_L1));
	double t = (double)cvGetTickCount();
	cvL1StOMPSolve( icvAOps, icvAtOps, A, Y, X0, epsilon );
	t = (double)cvGetTickCount() - t;
	printf( "time = %gms\n", t/((double)cvGetTickFrequency() * 1000.) );
	printf("||X0 - X|| After L1StOMP : %f\n", cvNorm(X0, X, CV_L1));
	cvReleaseMat( &X0 );
	cvReleaseMat( &e );
	cvReleaseMat( &Y );
	cvReleaseMat( &X );
	cvReleaseMat( &A );
	return 0;
}

