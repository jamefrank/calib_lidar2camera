#include "ba_calib.h"
#include "cmdline.h"
#include "utils.h"
#include <Eigen/src/Core/Matrix.h>
#include <cassert>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>

#include <string>
#include <vector>
using namespace std;

int main(int argc, char *argv[]) {
  spdlog::info("Welcome to use undist points tool!");

  cmdline::parser parser;
  parser.add<string>("file", 'f', "Input points txt", true, "");
  parser.add<string>("img", 'i', "Input img", false, "");
  parser.add<string>("config", 'c', "Input config file", true, "");
  parser.add<string>("camera-model", 't', "Camera model: pinhole|fisheye", false, "pinhole",
                     cmdline::oneof<string>("pinhole", "fisheye"));
  parser.add<string>("action-model", 'a', "Action model: undist|dist", false, "undist",
                     cmdline::oneof<string>("undist", "dist"));
  parser.add<string>("output", 'o', "Output directory containing undist/dist results",
                     true, "");
  parser.add("verbose", '\0', "verbose when undist or dist");

  parser.parse_check(argc, argv);

  // load points txt
  std::string points_file = parser.get<std::string>("file");
  std::vector<cv::Point2f> points = lidar2cam::utils::load_named_img_points(points_file);

  // process points
  std::string config_file = parser.get<std::string>("config");
  YAML::Node config = YAML::LoadFile(config_file);
  cv::Mat K, D;
  lidar2cam::utils::parse_K(config, "K", K);
  lidar2cam::utils::parse_D(config, "D", D);
  lidar2cam::utils::log_cvmat(K, "K");
  lidar2cam::utils::log_cvmat(D, "D");
  lidar2cam::CameraType camera_type;
  std::string cam_type = parser.get<std::string>("camera-model");
  if ("pinhole" == cam_type) {
    camera_type = lidar2cam::CameraType::PINHOLE;
  } else if ("fisheye" == cam_type) {
    camera_type = lidar2cam::CameraType::FISHEYE;
  } else {
    spdlog::error("unkonw camera type");
    exit(-1);
  }
  cv::Mat points_reulst;
  std::string action_modal = parser.get<std::string>("action-model"); 
  if("undist" == action_modal){
    lidar2cam::utils::undistPointsGeneral(cv::Mat(points), K, D, camera_type, points_reulst);
  }
  else if("dist" == action_modal){
    lidar2cam::utils::distPointsGeneral(cv::Mat(points), K, D, camera_type, points_reulst);
  }
  lidar2cam::utils::log_cvmat(points_reulst, "points_reulst");
  

  // // load img and draw (optional)
  // bool img_file = parser.exist("img");   //TODO
  // cv::Mat image;
  // if (img_file){
  //   std::string img_file = parser.get<std::string>("img");
  //   image = cv::imread(img_file, -1); 
  //   lidar2cam::utils::drawPoints(image, points_reulst);
  // }
 
  // output process results
  std::string outputDir = parser.get<std::string>("output");
  std::string out_txt = outputDir + "/" + ("undist"==action_modal?"undist.txt":"dist.txt");
  lidar2cam::utils::savePoints(points_reulst, out_txt);
  // if(img_file){
  //   std::string out_img = outputDir + "/" + ("undist"==action_modal?"undist.png":"dist.png");
  //   cv::imwrite(out_img, image);
  // }

  return 0;
}