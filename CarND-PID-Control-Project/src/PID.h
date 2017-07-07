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
};

#endif /* PID_H */
