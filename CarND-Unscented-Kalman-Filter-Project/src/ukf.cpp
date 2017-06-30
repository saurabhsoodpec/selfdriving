#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  
  n_x_ = 5;
  n_aug_ = 7;
  
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.4;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.5;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.01;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.25;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;
  MatrixXd H_laser;
  MatrixXd R_laser;
        	 
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    // first measurement
    x_ << 1, 1, 1, 1, 0;
	
	//TODO: Check what should be default value of P_
	P_ << 1, 0, 0, 0, 0,
		  0, 1, 0, 0, 0,
		  0, 0, 1, 0, 0,
		  0, 0, 0, 1, 0,
		  0, 0, 0, 0, 1;

	H_laser = MatrixXd(2, 5);
	H_laser << 1, 0, 0, 0, 0,
			   0, 1, 0, 0, 0;
			   
	R_laser = MatrixXd(2, 2);
  	R_laser << std_laspx_*std_laspx_, 0,
        	 0, std_laspy_*std_laspy_;
	
	/*
	P_ <<   0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;
    */
          	  
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      cout << "TODO:: U Kalman Filter Initialization RADAR" << endl;
      
      //Convert radar from polar to cartesian coordinates and initialize state.
	  float ro     = meas_package.raw_measurements_(0);
      float phi    = meas_package.raw_measurements_(1);
      float ro_dot = meas_package.raw_measurements_(2);
      x_(0) = ro     * cos(phi);
      x_(1) = ro     * sin(phi);      
      x_(2) = ro_dot * cos(phi);
      x_(3) = ro_dot * sin(phi);
      x_(4) = 0; // Initial Noise 
      

    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      	cout << "U Kalman Filter Initialization LASER" << endl;
		if ( fabs(meas_package.raw_measurements_[0]) < 0.001 && fabs(meas_package.raw_measurements_[1]) < 0.001 )  {
   			meas_package.raw_measurements_[0] = 0.001;   
   			meas_package.raw_measurements_[1] = 0.001;
		}
		//set the state with the initial location and zero velocity
		x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
		
    }
	
	time_us_ = meas_package.timestamp_;
	is_initialized_ = true;
    cout << "U Kalman Filter Initialization COMPLETE" << endl;
    return;
  }
  
  	//compute the time elapsed between the current and previous measurements
	double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
	time_us_ = meas_package.timestamp_;
    
   if(use_laser_ || use_radar_) {
   		cout << "Preparing for PREDICTION" << endl;
		while (delta_t > 0.2) {
      		double step = 0.1;
      		Prediction(step);
      		delta_t -= step;
  		}
  		Prediction(delta_t);
  	}
  
  
    if(use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    	cout << "IN LASER" << endl;
    	
    	UpdateLidar(meas_package.raw_measurements_);
    } 
    
    if(use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    	cout << "IN RADAR" << endl;
    	
    	UpdateRadar(meas_package.raw_measurements_);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  cout << "************************* Prediction method START *************************" << endl;
  AugmentedSigmaPoints(&Xsig_aug);
  //cout << "1 Xsig_aug = " << Xsig_aug << endl;
  SigmaPointPrediction(Xsig_aug, delta_t, &Xsig_pred_);
  //cout << "2 Xsig_pred_ = " << Xsig_pred_ << endl;
  PredictMeanAndCovariance(Xsig_pred_);
  //cout << "x_ = " << x_ << "P_ = " << P_ << endl;
  cout << "************************* Prediction method END *************************" << endl;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const VectorXd &z) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  
  //
  cout << "************************* UpdateLidar method START *************************" << endl;
  cout << "UpdateLidar UpdateLidar start. " << "x_ = " << endl << x_ << endl <<  "P_ = " << endl << P_ << endl << " and H_laser = " << H_laser << endl;
  
  //measurement matrix
        		
  VectorXd z_pred = H_laser * x_;
  cout << "UpdateLidar z_pred=" << z_pred << endl;
  VectorXd y = z - z_pred;
  UpdateMeasurement(y, H_laser, R_laser);
  cout << "************************* UpdateLidar method END *************************" << endl;
}

void UKF::UpdateMeasurement(const VectorXd &y, MatrixXd H_, MatrixXd R_) {

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
	//cout << "UpdateMeasurement x_=" << x_ << endl;
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	//cout << "UpdateMeasurement I=" << I << endl;
	P_ = (I - K * H_) * P_;
	//cout << "UpdateMeasurement Update end. P_=" << P_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const VectorXd &z) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  cout << "************************* UpdateRadar method START *************************" << endl;
  VectorXd z_pred;
  MatrixXd S;
  MatrixXd Zsig;
  //VectorXd z = meas_package.raw_measurements_;
  PredictRadarMeasurement(Xsig_pred_, &z_pred, &S, &Zsig);
  cout << "************************* UpdateRadar - PredictRadarMeasurement method END *************************" << endl;
  cout << "************************* UpdateRadar - UpdateState method TRIGGER START *************************" << endl;
  UpdateState(Xsig_pred_, Zsig, S, z_pred, z);
  cout << "************************* UpdateRadar - UpdateState method END *************************" << endl;
  
}


void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) { 
  cout << "************************* AugmentedSigmaPoints method START *************************" << endl;
  //define spreading parameter
  lambda_ = 3 - n_aug_;
  
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  
/*******************************************************************************
 * Student part begin
 ******************************************************************************/
 
  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++) {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }
  
/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  std::cout << "AugmentedSigmaPoints -> Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  //write result
  *Xsig_out = Xsig_aug;
   cout << "************************* AugmentedSigmaPoints method END *************************" << endl;
}

void UKF::SigmaPointPrediction(MatrixXd Xsig_aug, double delta_t, MatrixXd* Xsig_out) {

cout << "************************* SigmaPointPrediction method START *************************" << endl;
  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    //NormalizeAngle(&Xsig_pred_(3,i));
    Xsig_pred(4,i) = yawd_p;
  }

/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  //std::cout << "SigmaPointPrediction OUT -> Xsig_pred = " << std::endl << Xsig_out << std::endl;

  //write result
  *Xsig_out = Xsig_pred;
	cout << "************************* SigmaPointPrediction method END *************************" << endl;
}

void UKF::PredictMeanAndCovariance(MatrixXd Xsig_pred) {

  //define spreading parameter
  lambda_ = 3 - n_aug_;

  //create vector for weights
  weights_ = VectorXd(2*n_aug_+1);
  
/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  // set weights
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }
  
  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred.col(i);
  }

  //cout << "PredictMeanAndCovariance - 3" <<endl;

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    NormalizeAngle(&Xsig_pred(3,i));
    NormalizeAngle(&x_(3));
    VectorXd x_diff = Xsig_pred.col(i) - x_;
    //cout << "x_diff=" << x_diff << endl << "x_diff(3) =" << x_diff(3) << endl;
     
    //angle normalization
	while (x_diff(3)> M_PI) { x_diff(3)-=2.*M_PI;  } //cout << "PredictMeanAndCovariance - 3.3, x_diff(3)=" << x_diff(3) <<endl; 
    while (x_diff(3)<-M_PI) { x_diff(3)+=2.*M_PI;  } //cout << "PredictMeanAndCovariance - 3.6, x_diff(3)="<< x_diff(3) <<endl;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }

  //cout << "PredictMeanAndCovariance - 4" <<endl;
/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  //std::cout << "Predicted state" << std::endl;
  //std::cout << x_ << std::endl;
  //std::cout << "Predicted covariance matrix" << std::endl;
  //std::cout << P_ << std::endl;
  
}

void UKF::PredictRadarMeasurement(MatrixXd Xsig_pred, VectorXd* z_out, MatrixXd* S_out, MatrixXd* Zsig_out) {

  //cout << "************************* PredictRadarMeasurement method START *************************" << endl;

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //define spreading parameter
  double lambda = 3 - n_aug_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  //cout << "*************************1 PredictRadarMeasurement method START *************************" << endl;

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred(0,i);
    double p_y = Xsig_pred(1,i);
    double v  = Xsig_pred(2,i);
    double yaw = Xsig_pred(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;
	
	double param = p_x*p_x + p_y*p_y;
	if(fabs(param) < 0.001) {
		param = 0.001;
	}
    // measurement model
    Zsig(0,i) = sqrt(param);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(param);   //r_dot
  }

  //cout << "*************************2 PredictRadarMeasurement method START *************************" << endl;

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_ +1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //cout << "*************************3 PredictRadarMeasurement method START *************************" << endl;

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //cout << "*************************4 PredictRadarMeasurement method START *************************" << endl;
  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S = S + R;
  
/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //cout << "*************************5 PredictRadarMeasurement method START *************************" << endl;

  //print result
  //std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  //std::cout << "S: " << std::endl << S << std::endl;

  //write result
  *z_out = z_pred;
  *S_out = S;
  *Zsig_out = Zsig;

   //cout << "*************************6 PredictRadarMeasurement method START *************************" << endl;
}

void UKF::UpdateState(MatrixXd Xsig_pred, MatrixXd Zsig, MatrixXd S, VectorXd z_pred, VectorXd z) {

  //cout << "************************* UpdateState method START *************************" << endl;
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //define spreading parameter
  lambda_ = 3 - n_aug_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  //cout << "*************************1 UpdateState method START *************************" << endl;

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points


	NormalizeAngle(&Zsig(1, i));
    NormalizeAngle(&z_pred(1));

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    cout << "M_PI=" << M_PI << endl;
	cout << "z_diff=" << z_diff << endl;
	
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    //cout << "*************************1.5 UpdateState method START *************************" << endl;

	NormalizeAngle(&Xsig_pred(3,i));
    NormalizeAngle(&x_(3));
    
    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x_;
	cout << "x_diff=" << x_diff << endl;
	

    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //cout << "*************************2 UpdateState method START *************************" << endl; 

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  //cout << "*************************3 UpdateState method START *************************" << " z_pred=" << z_pred << "z="<< z << endl;
  //residual
  VectorXd z_diff = z - z_pred;

  //cout << "*************************4 UpdateState method START *************************" << endl;
  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //cout << "*************************5 UpdateState method START *************************" << endl;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  //std::cout << "Updated state x: " << std::endl << x_ << std::endl;
  //std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;

}

/**
 * Normalizes a given double angle between -Pi to Pi
 * @param {measurementPackage} pValue: Variable to be normalized
 */
void UKF::NormalizeAngle(double *pValue) {
  if (fabs(*pValue) > M_PI) {
  	//cout << "Doing Normalization ..." << *pValue << endl;
    *pValue -= round(*pValue / (2.0 * M_PI)) * (2.0 * M_PI);
    //cout << "After Normalization ..." << *pValue << endl;
  }
}