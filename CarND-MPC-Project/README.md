# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

---
## Data Preparation

1. We receive the all the input points in MAP co-ordinates, we need to convert them to vehicle coordinates.
2. Use the input parameters to fit the polynomial and calculate the reference line. This will be the Green line in the final video.
3. Create the current state object with the vehicle parameters.
4. Evaluate the new state using the current state, including the latency of the 100ms.
5. Find the best fit steering and throttle by using IOPT SOLVE operation.

## Handling Control Latency 
In real driving situations there is a latency between when the actuation command is given and when the actuation actually happens. This is due to the various systems involved from trigger to implementation. In this project Latency is handled during the initial calculation of the car state. Latency parameter defined in this model is 100ms.
The approach to handle latency is to evaluate the future state of the car after taking into consideration the latency (equations shown below). 

			/* Form the new state, considering the latency delay */
		    const double dMultiplier = (dSteer / Lf); 
		    const double x0_new = 0.0 + (v * cos(0.0) * LATENCY);  /* X, Y and Psi are 0 in vehicle frame */
		    const double y0_new = 0.0 + (v * sin(0.0) * LATENCY);
		    const double psi0_new = 0.0 - (v * dMultiplier * LATENCY);
		    const double v0_new = v + (dThrottle * LATENCY); /* Assume throttle to be acceleration */
		    const double cte0_new = cte - (v * sin(epsi) * LATENCY);
		    const double epsi0_new = epsi - (v * dMultiplier * LATENCY);
		    state << x0_new, y0_new, psi0_new, v0_new, cte0_new, epsi0_new;

Now using this new state as "x0" means that car has driven to the new state for the time duration of the LATENCY (100ms). Now this new state is the input state and the next set of actuations will be applied on this new updated state.


## The Model
Model used in the project is based on MPC i.e. Model Predictive Control - Plot a Reference trajectory using the given waypoints, get the initial state and fitted polynomial as the input to a IOPT Solver. Solver uses the Model, Constrains and Cost function to evaluate the optimized delta (steering) and throttle (acceleration). Following are the state params - 

1. [x,y] - The co-ordinates of the vehicle in vehicle co-ordinates.
2. [psi] - The heading of the vehicle
3. [v] - The velocity of the vehicle
4. [cte] - The cross track error. This is calculated based on the distance from the reference trajectory i.e. the middle of the road
5. [ephi] - The heading error (i.e. the difference between ideal heading and actual heading)

Variable Constraints are defined as  below. Here the upper and lower value of the constraint is defined for  
	
	...
	// TODO: Set lower and upper limits for variables.
	for(int i=0; i< n_constraints; i++) {
		vars_lowerbound[i] = -1.0e19;
		vars_upperbound[i] = 1.0e19;
	}
	for(int i=0; i< (N-1); i++) {
		vars_lowerbound[delta_start + i] = -0.436332; // (-25 degree in radians) //  * Lf
		vars_upperbound[delta_start + i] = 0.436332;
		vars_lowerbound[a_start + i] = -1.0;
		vars_upperbound[a_start + i] = 1.0;
	  }
	// Lower and upper limits for the constraints 
	// Should be 0 besides initial state.
	Dvector constraints_lowerbound(n_constraints);
	Dvector constraints_upperbound(n_constraints);
	for (int i = 0; i < n_constraints; i++) {
		constraints_lowerbound[i] = 0;
		constraints_upperbound[i] = 0;
	}

	constraints_lowerbound[x_start] = x;
	constraints_lowerbound[y_start] = y;
	constraints_lowerbound[psi_start] = psi;
	constraints_lowerbound[v_start] = v;
	constraints_lowerbound[cte_start] = cte;
	constraints_lowerbound[epsi_start] = epsi;

	constraints_upperbound[x_start] = x;
	constraints_upperbound[y_start] = y;
	constraints_upperbound[psi_start] = psi;
	constraints_upperbound[v_start] = v;
	constraints_upperbound[cte_start] = cte;
	constraints_upperbound[epsi_start] = epsi;

## COST Function is calculated by adding all the errors with respective weights - 

	for(int i=0; i<N; i++) {
		  fg[0] += 100 * CppAD::pow(vars[cte_start+i],2);
		  fg[0] += 100 * CppAD::pow(vars[epsi_start+i],2);
		  fg[0] += 0.1 * CppAD::pow(vars[v_start+i] - v_ref ,2);
	  }
	  for(int i=0; i<N-1; i++) {
		  fg[0] += CppAD::pow(vars[delta_start+i],2);
		  fg[0] += CppAD::pow(vars[a_start+i],2);
	  }
	  for(int i=0; i<N-2; i++) {
		  fg[0] += 5000 * CppAD::pow(vars[delta_start+i+1] - vars[delta_start+i],2);
		  fg[0] += CppAD::pow(vars[a_start+i+1] - vars[a_start+i],2);
	  }

##  Hyper Parameters
The hyper parameters (the weights of the cost function, N and dt) were chosen based on the fact that the accuracy of the car is maintained if we keep "dt" small and N will provide by visibility for the number of iterations. The totalT i.e. N * dt is 5 seconds. This means the model predicts upto five seconds into the future. This keep the model limited and accurate. Also reference velocity is decided based on the expected speed of the car and accuracy of driving.
Here are the final parameters - 

1. N = 10;
2. dt = 0.05;
3. v_ref = 90; // ref velocity

Also, weights of the various cost prameters are tuned to achieve smooth driving experience. Here are the final weights - 
1. CTE Weight = 100 i.e. ---- Allow CTE to play an important role and penalize the car if it goes off from the center.
2. EPSI Weight = 100 i.e. ---- Keep car aligned to the expected reference orientation.
3. Velocity Weight = 0.1 -- This is required so that Car drives smoothly and is able to pick up in a smooth way. 
4. Steering Weight = 1 -- Weight to keep driving in the correct direction.
5. Acceleration Weight  =1 -- Weight to increase the speed if possible. This is not very important as other factors are more important, but this is required to motivate the model to increase speed.
6. Change in Steering Weight = 5000 --- To really discourage rapid change in angle.
7. Change in acceleration = 1  -- Discourage rapid increase in speed.

## Constraints
Here are the constrain equations - 
<pre>

	      AD<double> f0 = coeffs[0] + coeffs[1] * x0 + coeffs[2]*x0*x0 + coeffs[3]*x0*x0*x0;
	      AD<double> psides0 = CppAD::atan(coeffs[1]+2*coeffs[2]*x0 + 3 * coeffs[3]*x0*x0);
	      fg[1 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
	      fg[1 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
	      fg[1 + psi_start + t] = psi1 - (psi0 - v0 * delta0 / Lf * dt); //psi + v / Lf * delta * dt;
	    	  fg[1 + v_start + t] = v1 - (v0 + a0 * dt);  //v + a * dt;
	    	  fg[1 + cte_start + t] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
	    	  fg[1 + epsi_start + t] = epsi1 - ((psi0 - psides0) - v0 * delta0 / Lf * dt);
</pre>

## Final solution
Final solution video is available here - (https://youtu.be/UTE-S2ECbng)

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Fortran Compiler
  * Mac: `brew install gcc` (might not be required)
  * Linux: `sudo apt-get install gfortran`. Additionall you have also have to install gcc and g++, `sudo apt-get install gcc g++`. Look in [this Dockerfile](https://github.com/udacity/CarND-MPC-Quizzes/blob/master/Dockerfile) for more info.
* [Ipopt](https://projects.coin-or.org/Ipopt)
  * Mac: `brew install ipopt`
  * Linux
    * You will need a version of Ipopt 3.12.1 or higher. The version available through `apt-get` is 3.11.x. If you can get that version to work great but if not there's a script `install_ipopt.sh` that will install Ipopt. You just need to download the source from the Ipopt [releases page](https://www.coin-or.org/download/source/Ipopt/) or the [Github releases](https://github.com/coin-or/Ipopt/releases) page.
    * Then call `install_ipopt.sh` with the source directory as the first argument, ex: `bash install_ipopt.sh Ipopt-3.12.1`. 
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [CppAD](https://www.coin-or.org/CppAD/)
  * Mac: `brew install cppad`
  * Linux `sudo apt-get install cppad` or equivalent.
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions


1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.

## Tips

1. It's recommended to test the MPC on basic examples to see if your implementation behaves as desired. One possible example
is the vehicle starting offset of a straight line (reference). If the MPC implementation is correct, after some number of timesteps
(not too many) it should find and track the reference line.
2. The `lake_track_waypoints.csv` file has the waypoints of the lake track. You could use this to fit polynomials and points and see of how well your model tracks curve. NOTE: This file might be not completely in sync with the simulator so your solution should NOT depend on it.
3. For visualization this C++ [matplotlib wrapper](https://github.com/lava/matplotlib-cpp) could be helpful.
