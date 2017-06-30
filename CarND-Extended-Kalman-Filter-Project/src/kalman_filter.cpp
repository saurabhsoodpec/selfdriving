#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
	cout << "KalmanFilter Predict start. " << "x_ = " << x_ << " and F_ = " << F_ << endl;
	x_ = F_ * x_;
	cout << "KalmanFilter Updated X. "  << " x_ = " << x_ << endl;
	MatrixXd Ft = F_.transpose();
	cout << "KalmanFilter Created F transpose" << " Ft=" << Ft << endl;
	cout << "KalmanFilter P_=" << P_ << endl;
	cout << "KalmanFilter Q_=" << Q_ << endl;
	P_ = F_ * P_ * Ft + Q_;
	cout << "KalmanFilter Predict end" << " P_=" << P_ << endl;
}

void KalmanFilter::Update(const VectorXd &z) {
	//cout << "KalmanFilter Update start. " << "x_ = " << x_ << " and H_ = " << H_ << endl;
	VectorXd z_pred = H_ * x_;
	//cout << "KalmanFilter z_pred=" << z_pred << endl;
	VectorXd y = z - z_pred;
	UpdateMeasurement(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    cout << "KalmanFilter UpdateEKF start. " << "x_ = " << x_ << " and H_ = " << H_ << endl;
    cout << "KalmanFilter z=" << z << endl;	
	
	float px = x_(0u);
	float py = x_(1u);
	float vx = x_(2u);
	float vy = x_(3u);
	cout << "KalmanFilter px=" << px << endl;
	
	//pre-compute a set of terms 
	float norm = sqrt(px*px+py*py);
	Eigen::MatrixXd hx_ = MatrixXd(3, 1);
	
	if (norm > 0.00001f) {
		float c1 = atan2(py,px);
		float c2 = ((px*vx+py*vy)/norm);
		cout << "KalmanFilter c2=" << c2 << endl;
		
		hx_ << norm,
			  c1,
			  c2;
		cout << "KalmanFilter hx=" << hx_ << endl;
		
	} else {
		cout << "KalmanFilter with ZERO NORM. Using Jacobian for measurement update." << endl;
		//Fallback to only Jacobian matrix 
		hx_ = (H_ * x_);
		cout << "KalmanFilter Jacobian hx=" << hx_ << endl;
	}
	
	//Applying measurement update.
	VectorXd y = z - hx_;
	cout << "KalmanFilter y=" << y << endl;
	// angle normalization
	while (y(1)> M_PI) y(1)-=2.*M_PI;
	while (y(1)<-M_PI) y(1)+=2.*M_PI;
	cout << "KalmanFilter After normalization y=" << y << endl;

	UpdateMeasurement(y);
}

void KalmanFilter::UpdateMeasurement(const VectorXd &y) {
	MatrixXd Ht = H_.transpose();
	//cout << "KalmanFilter Ht=" << Ht << endl;
	//cout << "KalmanFilter R_=" << R_ << endl;
	MatrixXd S = H_ * P_ * Ht + R_;
	//cout << "KalmanFilter S=" << S << endl;
	MatrixXd Si = S.inverse();
	//cout << "KalmanFilter Si=" << Si << endl;
	MatrixXd PHt = P_ * Ht;
	//cout << "KalmanFilter PHt=" << PHt << endl;
	MatrixXd K = PHt * Si;
	//cout << "KalmanFilter K=" << K << endl;
	//new estimate
	x_ = x_ + (K * y);
	cout << "KalmanFilter x_=" << x_ << endl;
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	cout << "KalmanFilter I=" << I << endl;
	P_ = (I - K * H_) * P_;
	cout << "KalmanFilter Update end. P_=" << P_ << endl;
}
