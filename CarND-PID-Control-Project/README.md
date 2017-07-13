# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

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
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

There's an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 

## Evaluating PID values
Initially I started with a default PID values and tuned those values manually to reach a stable configuration where the car was driving optimally and car was following the track for some time. Then I implemented dynamic TWIDDLE algorithm which optimizes PID values while driving the car and tries to reach a better and more stable configuration after each run. The approach for TWIDDLE is to tune each one parameter at a time and check if the changed helped to reduce the total error. If the error is reduced in the next iteration then the change is kept else the change is ignored.

Here is the TWIDDLE algorithm - 

def twiddle(tol=0.2): 
    p = [0, 0, 0]
    dp = [1, 1, 1]
    robot = make_robot()
    x_trajectory, y_trajectory, best_err = run(robot, p)

    it = 0
    while sum(dp) > tol:
        print("Iteration {}, best error = {}".format(it, best_err))
        for i in range(len(p)):
            p[i] += dp[i]
            robot = make_robot()
            x_trajectory, y_trajectory, err = run(robot, p)

            if err < best_err:
                best_err = err
                dp[i] *= 1.1
            else:
                p[i] -= 2 * dp[i]
                robot = make_robot()
                x_trajectory, y_trajectory, err = run(robot, p)

                if err < best_err:
                    best_err = err
                    dp[i] *= 1.1
                else:
                    p[i] += dp[i]
                    dp[i] *= 0.9
        it += 1
    return p
    
 The code is customized to adapt to dynamic driving scenario in the method PID.twiddle()
 https://github.com/saurabhsoodpec/selfdriving/blob/master/CarND-PID-Control-Project/src/PID.cpp  
 
 Also, the Throttle was customized based on the steer angle. Refer to the method PID.getThrottle().
 
 ## Here are the final PID values -

Kp = 0.14
Ki = 0
Kd = 3.16886

These are the values used as initial default for the program, but since dynamic twiddle is enable, these values are modified at the runtime to achive a even better config based on other parameters such as track and driving speed.

You can check the final result in the video - https://youtu.be/0-3Hd-T9rg4
