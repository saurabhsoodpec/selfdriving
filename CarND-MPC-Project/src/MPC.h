#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

class MPC {
 public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);

  /*!
   * Gets the latest X and Y trajectories
   */
  vector<double> getXs();
  vector<double> getYs();

   private:
    vector<double> solutionsArr;

};

#endif /* MPC_H */
