#include "cvl1qc.h"

void icvAOps( CvMat* X, CvMat* Y, void* userdata )
{
	CvMat* A = (CvMat*)userdata;
	cvMatMul( A, X, Y );
}

void icvAtOps( CvMat* X, CvMat* Y, void* userdata )
{
	CvMat* A = (CvMat*)userdata;
	cvGEMM( A, X, 1, NULL, 0, Y, CV_GEMM_A_T );
}

int main()
{
	const int N = 4096;
	const int K = 64;
	const int T = 10;
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
	double sigma = .005;
	CvMat* e = cvCreateMat( K, 1, CV_64FC1 );
	cvRandArr( &rng_state, e, CV_RAND_NORMAL, cvScalar(0), cvScalar(sigma) );
	cvMatMulAdd( A, X, e, Y );
	double epsilon = sigma * sqrt(K) * sqrt(1 + 2 * sqrt(2) / sqrt(K));
	CvMat* X0 = cvCreateMat( N, 1, CV_64FC1 );
	printf("||X0 - X|| Before L1QC : %f\n", cvNorm(X0, X, CV_L1));
	cvL1QCSolve( icvAOps, icvAtOps, A, Y, X0, epsilon );
	printf("||X0 - X|| After L1QC : %f\n", cvNorm(X0, X, CV_L1));
	//for ( int i = 0; i < N; i++ )
	//	printf("%f %f\n", X->data.db[i], X0->data.db[i]);
	cvReleaseMat( &X0 );
	cvReleaseMat( &e );
	cvReleaseMat( &Y );
	cvReleaseMat( &X );
	cvReleaseMat( &A );
	return 0;
}
