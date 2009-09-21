#ifndef _GUARD_cvl1stomp_h_
#define _GUARD_cvl1stomp_h_

#include <cv.h>

typedef void (*CvSparseMatOps) ( CvMat*, CvMat*, CvMat*, void* );

int cvL1StOMPSolve( CvMat* A, CvMat* B, CvMat* X, double epsilon, CvTermCriteria so_term_crit = cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 1e-5 ), CvTermCriteria cg_term_crit = cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 200, 1e-16 ) );

int cvL1StOMPSolve( CvSparseMatOps AOps, CvSparseMatOps AtOps, void* userdata, CvMat* B, CvMat* X, double epsilon, CvTermCriteria so_term_crit = cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 1e-5 ), CvTermCriteria cg_term_crit = cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 200, 1e-16 ) );

#endif
