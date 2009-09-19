#include "cvl1eq.h"

int cvL1EQSolve( CvMat* A, CvMat* B, CvMat* X, CvTermCriteria pd_term_crit, CvTermCriteria cg_term_crit )
{
	const double alpha = .01;
	const double beta = .5;
	const double mu = 10.;

	CvMat* AAt = cvCreateMat( A->rows, A->rows, CV_MAT_TYPE(A->type) );
	cvGEMM( A, A, 1, NULL, 0, AAt, CV_GEMM_B_T );
	CvMat* W = cvCreateMat( A->rows, 1, CV_MAT_TYPE(X->type) );
	if ( cvCGSolve( AAt, B, W, cg_term_crit ) > .5 )
	{
		cvReleaseMat( &W );
		cvReleaseMat( &AAt );
		return -1;
	}
	cvGEMM( A, W, 1, NULL, 0, X, CV_GEMM_A_T );
	cvReleaseMat( &W );
	cvReleaseMat( &AAt );

	CvMat* U = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );
	cvAbsDiffS( X, U, cvScalar(0) );
	CvScalar sumAbsX = cvSum( U );
	double minAbsX, maxAbsX;
	cvMinMaxLoc( U, &minAbsX, &maxAbsX );
	cvConvertScale( U, U, .95, maxAbsX * .1 );

	CvMat* fu1 = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );
	CvMat* fu2 = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );
	CvMat* lamu1 = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );
	CvMat* lamu2 = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );
	CvMat* v = cvCreateMat( B->rows, B->cols, CV_MAT_TYPE(B->type) );
	CvMat* atv = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );
	CvMat* rpri = cvCreateMat( B->rows, B->cols, CV_MAT_TYPE(B->type) );

	cvSub( X, U, fu1 );
	cvAddWeighted( X, -1, U, -1, 0, fu2 );
	cvDiv( NULL, fu1, lamu1 );
	cvDiv( NULL, fu2, lamu2 );
	cvSub( lamu2, lamu1, atv );
	cvMatMul( A, atv, v );
	cvGEMM( A, v, 1, NULL, 0, 0, atv, CV_GEMM_A_T );
	cvGEMM( A, X, 1, B, -1, 0, rpri );

	double sdg = 2.;
	double tau = mu * 2. / sdg;
	double tau_inv = 1. / tau;
	CvMat* rcent1 = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );
	CvMat* rcent2 = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );
	CvMat* rdual1 = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );
	CvMat* rdual2 = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );
	cvSet( rcent1, cvScalar(1. - tau_inv) );
	cvSet( rcent2, cvScalar(1. - tau_inv) );
	cvSub( lamu1, lamu2, rdual1 );
	cvAddWeighted( atv, 1, rdual1, 1, 1, rdual1 );
	cvAddWeighted( lamu1, -1, lamu2, -1, 2, rdual2 );
	double resnorm = cvNorm(rcent1) + cvNorm(rcent2) + cvNorm(rdual1) + cvNorm(rdual2) + cvNorm(rpri);

	for ( int pditer = 0; pditer < pd_term_crit.max_iter; ++pditer )
	{
		CvMat* w1 = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );
		CvMat* w2 = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );
		CvMat* w3 = cvCreateMat( B->rows, B->cols, CV_MAT_TYPE(B->type) );
		CvMat* w12 = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );
		CvMat* sig1 = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );
		CvMat* sig2 = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );
		CvMat* sigx = cvCreateMat( X->rows, X->cols, CV_MAT_TYPE(X->type) );

		double* fu1p = fu1->data.db;
		double* fu2p = fu2->data.db;
		double* atvp = atv->data.db;
		double* lamu1p = lamu1->data.db;
		double* lamu2p = lamu2->data.db;
		double* w1p = w1->data.db;
		double* w2p = w2->data.db;
		double* w12p = w12->data.db;
		double* sig1p = sig1->data.db;
		double* sig2p = sig2->data.db;
		double* sigxp = sigx->data.db;
		for ( int i = 0; i < X->rows; ++i, ++fu1p, ++fu2p, ++atvp, ++lamu1p, ++lamu2p, ++w1p, ++w2p, ++w12p, ++sig1p, ++sig2p, ++sigxp )
		{
			*w1p = tau_inv * (1. / (*fu1p) + 1. / (*fu2p)) - (*atvp);
			*w2p = -1 - tau_inv * (1. / (*fu1p) + 1. / (*fu2p));
			*sig1p = -(*lamu1p) / (*fu1p) - (*lamu2p) / (*fu2p);
			*sig2p = (*lamu1p) / (*fu1p) - (*lamu2p) / (*fu2p);
			*sigxp = (*sig1p) - (*sig2p) * (*sig2p) / (*sig1p);
			*w12p = (*w1p) / (*sigxp) - (*w2p) * (*sig2p) / ((*sigxp) * (*sig1p));
		}
		cvSubRS( rpri, cvScalar(0), w3 );

		CvMat* w1t = cvCreateMat( B->rows, B->cols, CV_MAT_TYPE(B->type) );
		cvMatMul( A, w12, w1t );
		cvSub( w1t, w3, w1t );

	}

__clean_up__:

	cvReleaseMat( &rdual2 );
	cvReleaseMat( &rdual1 );
	cvReleaseMat( &rcent2 );
	cvReleaseMat( &rcent1 );
	cvReleaseMat( &rpri );
	cvReleaseMat( &atv );
	cvReleaseMat( &v );
	cvReleaseMat( &lamu2 );
	cvReleaseMat( &lamu1 );
	cvReleaseMat( &fu2 );
	cvReleaseMat( &fu1 );
	cvReleaseMat( &U );
}
