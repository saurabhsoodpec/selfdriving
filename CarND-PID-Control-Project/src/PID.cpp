#include "PID.h"
#include <math.h>
#include <iostream>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {

}

PID::~PID() {}

void PID::Init() {
	Init(error_coefficients[0], error_coefficients[1], error_coefficients[2]);
}

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
	if (fabs(steer_value) > 0.3) {
		throttle_value = 0.1;
	} else if (fabs(steer_value) > 0.2) {
		throttle_value = 0.3;
	} else if (fabs(steer_value) > 0.1) {
		throttle_value = 0.4;
	}
	return throttle_value;
}

void PID::twiddle(double cte) {
	double previous_run_error = TotalError();
	bool is_first_m = is_first_measurement;
	cout << endl << "previous_run_error=" << previous_run_error << "  Prev_Coeffs=(" << error_coefficients[0] << "," << error_coefficients[1] << "," << error_coefficients[2] << ")" << " twiddle_state=" << twiddle_state << " tw_coeff_idx=" << tw_coeff_idx << endl;
	UpdateError(cte);
	double new_total_error = TotalError();
	if(is_first_m) {
		cout << "FIRST MEASUREMENT - NO UPDATE" << endl;
		best_error = new_total_error;
		return;
	}

	bool is_better = (fabs(new_total_error) < fabs(best_error)) ? true:false;

	if(is_better) {
		cout<< "BETTER ERROR ACHIEVED - OLD Coeff" << " Ks=(" << Kp << "," << Ki << "," << Kd << ")" << endl;
		Kp = error_coefficients[0];
		Ki = error_coefficients[1];
		Kd = error_coefficients[2];
		cout<< "BETTER ERROR ACHIEVED - NEW Coeff" << " Ks=(" << Kp << "," << Ki << "," << Kd << ")" << endl;
	}

	double pd_sum = 0;
	for (int s = 0; s < sizeof(error_coefficients_differencial); s++) {
		pd_sum += error_coefficients_differencial[s];
	}
	if (pd_sum > 0.02) {
		switch (twiddle_state) {
			case 0:
				error_coefficients[tw_coeff_idx] += error_coefficients_differencial[tw_coeff_idx];
				twiddle_state = 1;
				break;
			case 1:
				if (is_better) {
					best_error = new_total_error;
					error_coefficients_differencial[tw_coeff_idx] *= 1.1;
					tw_coeff_idx = (tw_coeff_idx + 1) % 3;
					twiddle_state = 0;
				} else {
					error_coefficients[tw_coeff_idx] -= 2 * error_coefficients_differencial[tw_coeff_idx];
					twiddle_state = 2;
				}
				break;
			case 2:
				if (is_better) {
					best_error = new_total_error;
					error_coefficients_differencial[tw_coeff_idx] *= 1.1;
				} else {
					error_coefficients[tw_coeff_idx] += error_coefficients_differencial[tw_coeff_idx];
					error_coefficients_differencial[tw_coeff_idx] *= 0.9;
				}
				tw_coeff_idx = (tw_coeff_idx + 1) % 3;
				twiddle_state = 0;
		}
	}
	cout << "new_total_error=" << new_total_error << " New_Coeffs=(" << error_coefficients[0] << "," << error_coefficients[1] << "," << error_coefficients[2] << ")" <<  " twiddle_state=" << twiddle_state << " tw_coeff_idx=" << tw_coeff_idx << endl << endl ;

}

