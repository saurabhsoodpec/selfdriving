#include "PID.h"
#include <math.h>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
	this->Kp = Kp;
	this->Ki = Ki;
	this->Kd = Kd;
	cte_prev = 0;
	is_first_measurement = true;
}

void PID::UpdateError(double cte) {
	if(is_first_measurement) {
		cte_prev = cte;
		is_first_measurement = false;
	}
	cte_hist.push_back(cte);
	while (cte_hist.size() > 100) {
		cte_hist.erase(cte_hist.begin());
	}
	double total_cte =0.0;
	for(int i = 0; i < cte_hist.size(); i++) {
		total_cte += cte_hist[i];
	}

	p_error = (-Kp) * cte;
	d_error = (-Kd) * (cte - cte_prev);
	i_error = (-Ki) *  total_cte;
	cte_prev = cte;

}

double PID::TotalError() {
	return (p_error + d_error + i_error);
}

double PID::getSteer() {
	double steer_value = TotalError();
	if (steer_value > 1.0) {
		steer_value = 1.0;
	} else if (steer_value < -1.0) {
		steer_value = -1.0;
	}
	return steer_value;
}

double PID::getThrottle() {
	double throttle_value = 0.5;

	double steer_value = getSteer();
	if (fabs(steer_value) > 0.1) {
		throttle_value = 0.0;
	} else if (fabs(steer_value) > 0.7) {
		throttle_value = 0.1;
	} else if (fabs(steer_value) > 0.3) {
		throttle_value = 0.3;
	}
	return throttle_value;
}


