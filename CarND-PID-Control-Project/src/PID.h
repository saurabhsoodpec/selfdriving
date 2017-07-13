#ifndef PID_H
#define PID_H

#include <vector>
using namespace std;

class PID {
public:
  /*
  * Errors
  */
  double p_error;
  double i_error;
  double d_error;
  double cte_prev;
  vector <double> cte_hist;
  bool is_first_measurement;

  double error_coefficients[3] = {0.14, 0, 3.16886}; //{0.155889,0.0,2.72039}; //{0.14, 3.25015e-05, 3.16886}; //{0.100847,3.25015e-05,3.16886}; //{0.0644,0.0,1.5}; // {0.0644, 0, 1.5};//{0.14, 0, 1.5}; //{0.08, 4.0, 0.04};
  double error_coefficients_differencial[3] = {2./3. *error_coefficients[0] , 0.001, 2./3. *error_coefficients[2]};
  double best_error;
  int twiddle_state =0;
  int tw_coeff_idx = 0;
  int noOfSteps = 0;
  /*
  * Coefficients
  */ 
  double Kp;
  double Ki;
  double Kd;

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  void Init();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();

  double getSteer();

  double getThrottle();

  void twiddle(double cte);
};

#endif /* PID_H */
