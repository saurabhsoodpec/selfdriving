#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2((map_y-y),(map_x-x));

	double angle = fabs(theta-heading);
  angle = min(2*pi() - angle, angle);

  if(angle > pi()/4)
  {
    closestWaypoint++;
  if (closestWaypoint == maps_x.size())
  {
    closestWaypoint = 0;
  }
  }

  return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};

}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;
  double max_car_speed = 50;
  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }

  int lane = 1;
  double ref_vel = 0;
  
  h.onMessage([&ref_vel, &map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy,&lane](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];


          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values 
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];

          	json msgJson;

          	// TODO: define a path made up of (x,y) points that the car will visit sequentially every .02 seconds

		  
          vector<double> ptsx;
          vector<double> ptsy;
          
          int path_size = previous_path_x.size();
          
          if(path_size > 0) {
			car_s = end_path_s;
		  }
		  
          double ref_x = car_x;
          double ref_y = car_y;
          double ref_yaw = deg2rad(car_yaw);
          
          //ref_vel = car_speed;
          
          bool too_close = false;
          bool turn_left_possible = true;
          bool turn_right_possible = true;
          double closestFrontCarVelocity = 0;
          float closestFrontCar_s = -1.0;
          for(int s_index=0; s_index<sensor_fusion.size(); s_index++) {
          	
          	float check_car_d = sensor_fusion[s_index][6];
          	float check_car_s = sensor_fusion[s_index][5];
          	float check_car_vx = sensor_fusion[s_index][3];
          	float check_car_vy = sensor_fusion[s_index][4];
          	float check_car_speed = sqrt(check_car_vx*check_car_vx + check_car_vy*check_car_vy);
          	
          	if(check_car_d > lane*4 && check_car_d < (lane+1)*4) {

          		if(check_car_s > car_s && (check_car_s-car_s) <30) {
          			cout << "********************** CHECK CAR SPEED  **** " << sensor_fusion[s_index][0] << " check_car_speed=" << check_car_speed << "\n"; 
          			too_close = true;
          			if(closestFrontCar_s == -1.0 || check_car_s < closestFrontCar_s) { // KEEP TRACK OF THE VELOCITY OF FRONT CAR
          				closestFrontCar_s = check_car_s;
          				closestFrontCarVelocity = check_car_speed;
          			}
          		}
          	} 
          	
          	if(turn_left_possible) {
          		if(check_car_d > ((lane-1)*4) && check_car_d < lane*4) {
          			if((check_car_s > car_s && (check_car_s-car_s) <30) || 
          			   (check_car_s < car_s && (car_s-check_car_s) <20)	||  //Add condition is speed of car behind is faster/slower.
          			   (check_car_s == car_s)) {
          				turn_left_possible = false;
          			} 
          		}
          	}
          	
          	if(turn_right_possible) {
          		if(check_car_d > ((lane+1)*4) && check_car_d < (lane+2)*4) {
          			if((check_car_s > car_s && (check_car_s-car_s) <30) || 
          			   (check_car_s < car_s && (car_s-check_car_s) <20)	||  //Add condition is speed of car behind is faster/slower.
          			   (check_car_s == car_s)) {
          				turn_right_possible = false;
          			} 
          		}
          	}
          	
          }
          
          if(too_close) {
          	cout << "ref_vel = " << ref_vel << "car_speed=" <<car_speed<<  "\n";
          	
          	if(lane>0 && turn_left_possible) {
          		cout << "********************** TURNING LEFT ****" << " lane=" << lane << "\n"; 
          		lane--;
          	} else if(lane<2 && turn_right_possible) {
          		cout << "********************** TURNING RIGHT ****" << " lane=" << lane << "\n"; 
          		lane++;
          	}
          	
          	if(closestFrontCarVelocity!=0 && ref_vel >= closestFrontCarVelocity) {  // ONLY REDUCE TO MATCH THE FRONT CAR
          		ref_vel -= 0.224;
          		cout << "********************** TOO CLOSE REDUCING ****" << " ref_vel=" << ref_vel << ", closestFrontCarVelocity="<<  closestFrontCarVelocity << "\n"; 
          	} else {
          		cout << "********************** NOT REDUCING ****" << " ref_vel=" << ref_vel << ", closestFrontCarVelocity="<<  closestFrontCarVelocity << "\n"; 
          	}
          	
          	
          } else if(ref_vel < 49.5) {
          
          	  cout << "ref_vel = " << ref_vel << "car_speed=" <<car_speed<<  "\n";
          	 ref_vel += 0.224;
          }
          
		  cout << "previous_path_x size=" << path_size << "\n";
			
          if(path_size < 2) {
          
          	  cout << "car_x,car_y,car_yaw =" << car_x << "," << car_y << "," << car_yaw << "\n";
          		
              double prev_pos_x = car_x - cos(car_yaw);
              double prev_pos_y = car_y - sin(car_yaw);
			  
              ptsx.push_back(prev_pos_x);
              ptsy.push_back(prev_pos_y);
              
              ptsx.push_back(car_x);
              ptsy.push_back(car_y);

          } else {
              ref_x = previous_path_x[path_size-1];
              ref_y = previous_path_y[path_size-1];

              double pos_x2 = previous_path_x[path_size-2];
              double pos_y2 = previous_path_y[path_size-2];
              ref_yaw = atan2(ref_y-pos_y2,ref_x-pos_x2);
  
              ptsx.push_back(pos_x2);
              ptsy.push_back(pos_y2);
                           
              ptsx.push_back(ref_x);
              ptsy.push_back(ref_y);
          }
		  //Convert the waypoint's co ordinates into S and D
		  //vector<double> point_frenet = getFrenet(pos_x, pos_y, angle, map_waypoints_x, map_waypoints_y);
		  vector<double> point_XY_30 = getXY(car_s+30, 2+4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
		  vector<double> point_XY_60 = getXY(car_s+60, 2+4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
		  vector<double> point_XY_90 = getXY(car_s+90, 2+4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
		  
          ptsx.push_back(point_XY_30[0]);
          ptsy.push_back(point_XY_30[1]);		  
          
          ptsx.push_back(point_XY_60[0]);
          ptsy.push_back(point_XY_60[1]);		
          
          ptsx.push_back(point_XY_90[0]);
          ptsy.push_back(point_XY_90[1]);		
		  
		  
		  //Convert points to car reference 
		  for (int i=0; i<ptsx.size(); i++) {
		  	//Shift car reference angle to 0 degrees
		  	double shift_x = ptsx[i]-ref_x;
		  	double shift_y = ptsy[i]-ref_y;
		  	
		  	ptsx[i] = shift_x*cos(0-ref_yaw) - shift_y*sin(0-ref_yaw);
		  	ptsy[i] = shift_x*sin(0-ref_yaw) + shift_y*cos(0-ref_yaw);
		  }
		  
		  tk::spline s;
		  s.set_points(ptsx, ptsy); 
		  
		  vector<double> next_x_vals;
          vector<double> next_y_vals;
           
          double dist_inc = 0.02 * ref_vel/2.24 ;
          double next_x = ref_x;
          double next_y = ref_y;
          
          double target_x = 30.0;
          double target_y = s(target_x);
          double target_dist = sqrt(target_x*target_x + target_y*target_y);
          double x_add_on = 0;
          
          
          for(int i = 0; i < 50; i++) {
          	  if (i<path_size) {
          	  	next_x = previous_path_x[i];
          	  	next_y = previous_path_y[i];
          	  } else {
          	    
          	  	//vector<double> current_SD= getFrenet(ref_x, ref_y, ref_yaw, map_waypoints_x, map_waypoints_y);
          	  	//double next_s = current_SD[0];
          	  	
          	  	double N = target_dist/dist_inc;
          	  	double x_point = x_add_on + target_x/N;
          	  	double y_point = s(x_point);
          	  	x_add_on = x_point;
          	  	
          	  	double x_point_ref = x_point;
          	  	double y_point_ref = y_point;
          	  	
          	  	//rotate back to normal 
          	  	x_point = x_point_ref * cos(ref_yaw) - y_point_ref*sin(ref_yaw);
          	  	y_point = x_point_ref * sin(ref_yaw) + y_point_ref*cos(ref_yaw);
          	  	
          	  	x_point += ref_x;
          	  	y_point += ref_y;
          	  	
          	  	next_x = x_point;
          	  	next_y = y_point;
          	  	cout << i << ":next_x=" << next_x << ",next_y=" << next_y << "\n"; 
              }
              next_x_vals.push_back(next_x);
              next_y_vals.push_back(next_y);
          }

          	//TODO:: END
          	
          	msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));
          	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);

        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
