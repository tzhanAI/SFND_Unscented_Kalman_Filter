#include <iostream>
#include <fstream>

#include "ukf.h"
#include "Eigen/Dense"

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  //time when the state is true, in us
  time_us_ = 0.0;

  //initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // set state dimension in measurement space
  n_z_ = 3;

  // set state dimension
  n_x_ = 5;

  // set augmented dimension, mu_a and mu_phidd are augmented
  n_aug_ = 7;

  // define spreading parameter. This impacts where the sigma points are
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.0;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  // Weights of sigma points size = 15
  weights_ = VectorXd(2 * n_aug_ + 1);

  double weight_0 = lambda_ / (lambda_ + n_aug_);
  double weight = 0.5 / (lambda_ + n_aug_);
  
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
	  weights_(i) = weight;
  }

  // create matrix for predicted sigma points 
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);

  // create matrix for sigma points in radar measurement space
  Zsig_ = MatrixXd(n_z_, 2 * n_aug_ + 1);
  Zsig_.fill(0.0);

  // predicted mean in radar measurement space
  z_pred_ = VectorXd(n_z_);
  z_pred_.fill(0.0);

  // predicted covariance matrix S radar measurement 
  S_ = MatrixXd(n_z_, n_z_);
  S_.fill(0.0);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_) {
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			/*
			Convert radar from polar to cartesian coordinates.
			Although radar gives velocity data in the form of the range rate rho_dot,
			a radar measurement does not contain enough information to determine the state
			variable velocities Vx and Vy. You can, however, use the radar measurements
			rho and phi to initialize the state variable locations Px and Py.
			*/
			double rho = meas_package.raw_measurements_(0);
			double phi = meas_package.raw_measurements_(1);
			double rho_dot = meas_package.raw_measurements_(2);
			x_ << cos(phi) * rho, sin(phi) * rho, 0.0, 0.0, 0.0;
			// init P_ to make it converge faster
			P_(0,0) = std_radr_ * std_radr_;		
			P_(1,1) = std_radr_ * std_radr_;
			P_(2,2) = std_radrd_ * std_radrd_;
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			double Px, Py;
			Px = meas_package.raw_measurements_(0);
			Py = meas_package.raw_measurements_(1);
			x_ << Px, Py, 0.0, 0.0, 0.0;
			// init P_ to make it converge faster
			P_(0,0) = std_laspx_ * std_laspx_;		
			P_(1,1) = std_laspy_ * std_laspy_;
		}
		
		//the time stamp extracted from first measurement data
		time_us_ = meas_package.timestamp_;
		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}

	/*****************************************************************************
	*  Prediction + Measurement Update
	****************************************************************************/
	//compute the time elapsed between the current and previous measurements
	double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
	time_us_ = meas_package.timestamp_;

	if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
	{
		use_radar_ == true;
		use_laser_ == false;
		//call UKF prediction step;
        Prediction(dt);
		//call UFK with input of z (rho, phi, rho_dot);
	    UpdateRadar(meas_package);
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::LASER) 
	{
		use_radar_ == false;
		use_laser_ == true;
		//call UKF prediction step;
        Prediction(dt);
		//call KF with input of z (Px, Py);
	    UpdateLidar(meas_package);
	}
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  /*****************************************************************************
   *  Create augmented mean state and sigma points matrix: x_aug_ and Xsig_aug_
   ****************************************************************************/
	//create augmented mean vector (7, 1) and covariance matrix (7, 7)
    VectorXd x_aug_ = VectorXd(n_aug_);
	MatrixXd P_aug_ = MatrixXd(n_aug_, n_aug_);

    x_aug_.fill(0.0);
    P_aug_.fill(0.0);

	x_aug_.head(n_x_) = x_;
	//The x_aug_(5) and x_aug_(6) are always zero in the augmented state vector;
	//The effects of std_a and std_yawdd are introduced from P_aug_;

	//create augmented covariance matrix
	P_aug_.topLeftCorner(n_x_, n_x_) = P_;
	P_aug_(5, 5) = std_a_ * std_a_;
	P_aug_(6, 6) = std_yawdd_ * std_yawdd_;

	//create square root matrix of P_aug_
	MatrixXd A = P_aug_.llt().matrixL();

	//create sigma point matrix (7, 2 * 7 + 1)
    MatrixXd Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    Xsig_aug_.fill(0.0);

	Xsig_aug_.col(0) = x_aug_;
	for (int i = 0; i < n_aug_; i++) {
		Xsig_aug_.col(i + 1) 		  = x_aug_ + sqrt(lambda_ + n_aug_) * A.col(i);
		Xsig_aug_.col(i + 1 + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * A.col(i);
	}
    
	/*****************************************************************************
	*  Predict sigma points : Xsig_pred_ using process model
	****************************************************************************/
    //create predicted sigma point matrix (5, 14)
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		double p_x =  Xsig_aug_(0, i);
		double p_y =  Xsig_aug_(1, i);
		double v   =  Xsig_aug_(2, i);
		double yaw =  Xsig_aug_(3, i);
		double yawd = Xsig_aug_(4, i);
		double nu_a = Xsig_aug_(5, i);
		double nu_yawdd = Xsig_aug_(6, i);

		//predicted state values px and py
		double px_p, py_p;
		double delta_t_2 = delta_t * delta_t;

		//avoid division by zero
		if (fabs(yawd) > 0.001) {
			px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
			py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
		}
		else {
			px_p = p_x + v * delta_t * cos(yaw);
			py_p = p_y + v * delta_t * sin(yaw);
		}

		//predicted state values continues... v, yaw and yawd
		double v_p = v;
		double yaw_p = yaw + yawd * delta_t;
		double yawd_p = yawd;

		//add noise (v and yaw accelaration's impacts)
		px_p = px_p + 0.5 * nu_a * delta_t_2 * cos(yaw);
		py_p = py_p + 0.5 * nu_a * delta_t_2 * sin(yaw);
		v_p = v_p + nu_a * delta_t;
		yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t_2;
		yawd_p = yawd_p + nu_yawdd * delta_t;

		//write predicted sigma point into right column
		Xsig_pred_(0, i) = px_p;
		Xsig_pred_(1, i) = py_p;
		Xsig_pred_(2, i) = v_p;
		Xsig_pred_(3, i) = yaw_p;
		Xsig_pred_(4, i) = yawd_p;
	}

	/*****************************************************************************
	*  Predict mean state x_ and P_ by using the predicted sigma points matrix Xsig_pred_
	****************************************************************************/
	//predicted state mean, clear out result from last loop before running sum;
	//new prediction sorely depends on new predicted sigma points Xsig_pred_;
	x_.fill(0.0);
	P_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		x_ += weights_(i) * Xsig_pred_.col(i);
	}

	//predicted state covariance matrix
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		// state difference									   
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		//angle normalization
		while (x_diff(3) > M_PI)  x_diff(3) -= 2.* M_PI;
		while (x_diff(3) < -M_PI) x_diff(3) += 2.* M_PI;

		P_ += weights_(i) * x_diff * x_diff.transpose();
	}
	
	/*****************************************************************************
	*  Predict mean state z_ and innovation matrix covariance S_ in Radar measurement 
	*  space by using predicted sigma points matrix Xsig_pred_, radar only step
	****************************************************************************/
	if (use_radar_ == true) {
		/**************************************************************
		* transform sigma points Xsig_pred_ into radar measurement space Zsig_(3, 15)
		***************************************************************/
		for (int i = 0; i < 2 * n_aug_ + 1; i++) {
			double Px =  Xsig_pred_(0, i);
			double Py =  Xsig_pred_(1, i);
			double v =   Xsig_pred_(2, i);
			double yaw = Xsig_pred_(3, i);

			double Px_2 = Px * Px;
			double Py_2 = Py * Py;

			double Rho = sqrt(Px_2 + Py_2); 
			double Phi = atan2(Py, Px);
			double Rhod = (Px * cos(yaw) * v + Py * sin(yaw) * v) / sqrt(Px_2 + Py_2);

			//assgin predicted sigma points in measurement space
			Zsig_(0, i) = Rho;
			Zsig_(1, i) = Phi;
			Zsig_(2, i) = Rhod;
		}

		/**************************************************************
		* calculate predicted mean z_pred in measurement space z using 
		* predicted sigma points matrix Zsig_
		***************************************************************/	
		//predicted state mean in measurement space z, clear out result from last loop before running sum;
	    //new prediction sorely depends on newly predicted sigma points in z space Zsig_;
		z_pred_.fill(0.0); 
		for (int i = 0; i < 2 * n_aug_ + 1; i++) {
			z_pred_ += weights_(i) * Zsig_.col(i);
		}

		//clear out result from last loop before running sum;
	    //new prediction sorely depends on newly predicted sigma points in z space Zsig_;
		S_.fill(0.0);
		// calculate innovation covariance matrix S
		for (int i = 0; i < 2 * n_aug_ + 1; i++) {
			// state difference
			VectorXd z_diff = Zsig_.col(i) - z_pred_;
			//angle normalization
			while (z_diff(1) > M_PI)  z_diff(1) -= 2.* M_PI;
			while (z_diff(1) < -M_PI) z_diff(1) += 2.* M_PI;

			S_ += weights_(i) * z_diff * z_diff.transpose();
		}

		// Add radar measurement noise R
		MatrixXd R = MatrixXd(n_z_, n_z_);
		R.fill(0.0);
		R(0, 0) = std_radr_ * std_radr_;
		R(1, 1) = std_radphi_ * std_radphi_;
		R(2, 2) = std_radrd_ * std_radrd_;
		S_ += R;
	}
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  // Calculate errors from prediction and measurements; x_(5,1)
   //Measurement matrix - lidar
	MatrixXd H_laser_(2, 5);
	H_laser_ << 1, 0, 0, 0, 0,
		        0, 1, 0, 0, 0;
	
	//Measurement noise matrix - lidar
	MatrixXd R_laser_(2, 2);
	R_laser_ << std_laspx_* std_laspx_, 0,
		        0, std_laspy_ * std_laspy_;

	//Mesaurement updates
	VectorXd y = meas_package.raw_measurements_ - H_laser_ * x_;
	// calculate Gain K and new estimates;
	MatrixXd Ht = H_laser_.transpose();
	MatrixXd PHt = P_ * Ht;
	MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
	MatrixXd Si = S.inverse();
	MatrixXd K = PHt * Si;

	//New estimate
	x_ = x_ + (K * y);
	int x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_laser_) * P_;

	// calculate NIS - Normalized Innovation Square (Consistency Check)
	double NIS_laser  = y.transpose() * Si * y;
	// print to files, NIS_laser
	ofstream myfile_NIS_laser ("NIS_laser.txt",  fstream::app);
	if (myfile_NIS_laser.is_open()){
		myfile_NIS_laser << NIS_laser << endl;
		myfile_NIS_laser.close();
	}
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // create matrix for cross correlation Tc
	MatrixXd Tc_ = MatrixXd(n_x_, n_z_);
    Tc_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		// x_ state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		// angle normalization
		while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
		while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

		// z_ state difference
		VectorXd z_diff = Zsig_.col(i) - z_pred_;
		// angle normalization
		while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
		while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

		Tc_ += weights_(i) * x_diff * z_diff.transpose();
	}

	// calculate Kalman gain K;
	MatrixXd K = Tc_ * S_.inverse();

	//error residual
	VectorXd z_diff = meas_package.raw_measurements_ - z_pred_;

	// angle normalization
	while (z_diff(1)> M_PI) z_diff(1) -= 2. * M_PI;
	while (z_diff(1)<-M_PI) z_diff(1) += 2. * M_PI;
	
	// update state mean and covariance matrix
	x_ += K * z_diff;
	P_ += -K * S_ * K.transpose();

	// calculate NIS - Normalized Innovation Square (Consistency Check)
	double NIS_radar  = z_diff.transpose() * S_.inverse() * z_diff;
	// print to files, NIS_laser
	ofstream myfile_NIS_radar ("NIS_radar.txt",  fstream::app);
	if (myfile_NIS_radar.is_open()){
		myfile_NIS_radar << NIS_radar << endl;
		myfile_NIS_radar.close();
	}
}
