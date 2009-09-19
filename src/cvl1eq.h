#ifndef _GUARD_cvl1eq_h_
#define _GUARD_cvl1eq_h_
#include <cv.h>
#include "cvcgsolve.h"

/* taken A, B, X, minimize ||X||_{L1} with constraint: AX = B */
int cvL1EQSolve( CvMat* A, CvMat* B, CvMat* X, CvTermCriteria pd_term_crit = cvTermCriteria( CV_TERMCRIT_EPS, 0, 1e-3 ), CvTermCriteria cg_term_crit = cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 200, 1e-16 ) );

#endif
