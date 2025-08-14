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
#include <spdlog/spdlog.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>

#include <vector>
using namespace std;

int main(int argc, char *argv[]) {
  spdlog::info("Welcome to use undist img tool!");

  cmdline::parser parser;
  parser.add<string>("input", 'i', "Input directory (example 'data')", true,
                     "");
  parser.add<string>("prefix", 'p', "Prefix of image name", false, "");
  parser.add<string>("extension", 'e', "File extension of images", false,
                     ".jpg");
  parser.add<int>(
      "board-width", 'w',
      "Number of inner corners on the charuco board pattern in x direction", false,
      3);
  parser.add<int>(
      "board-height", 'h',
      "Number of inner corners on the charuco board pattern in y direction", false,
      2);
  parser.add<double>("square-size", 's', "Size of one square (charuco) in mm", false,
                     60.0);
  parser.add<double>("marker-size", 'm', "Size of one square (charuco) in mm", false,
                     45.0);
  parser.add<string>("camera-model", 't', "Camera model", false, "pinhole",
                     cmdline::oneof<string>("pinhole", "fisheye"));
  parser.add<string>("action-model", 'a', "Action model", false, "undist",
                     cmdline::oneof<string>("undist", "dist"));
  parser.add<string>("output", 'o', "Output directory containing undist/dist results",
                     true, "");
  parser.add("detect", '\0', "detect when undist or dist");
  parser.add("verbose", '\0', "verbose when undist or dist");

  parser.parse_check(argc, argv);

  // load charuco imgs
  std::string inputDir = parser.get<std::string>("input");
  std::string prefix = parser.get<std::string>("prefix");
  std::string fileExtension = parser.get<std::string>("extension");
  std::vector<std::string> imageFilenames =
      lidar2cam::utils::get_file_lists(inputDir, prefix, fileExtension);
  if (imageFilenames.empty()) {
    spdlog::error("# ERROR: No chessboard images found.");
    return 1;
  } else {
    spdlog::info("# INFO: # images: {}", imageFilenames.size());
  }
  // detect charuco corners
  std::string config_file = inputDir + "/config.yaml";
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

  cv::Size boardSize;
  boardSize.width = parser.get<int>("board-width");
  boardSize.height = parser.get<int>("board-height");
  float squareSize = parser.get<double>("square-size");
  float markerSize = parser.get<double>("marker-size");
  cv::Mat objs = lidar2cam::utils::genObjs(boardSize, squareSize);

  bool verbose = parser.exist("verbose");
  bool detect = parser.exist("detect");
  std::string action_modal = parser.get<std::string>("action-model");
  std::string outputDir = parser.get<std::string>("output");


  for (int i = 0; i < imageFilenames.size(); i++) {
    cv::Mat image = cv::imread(imageFilenames[i], -1); 
    cv::Mat img;
    if("undist" == action_modal){
       lidar2cam::utils::undistImageGeneral(image, K, D, camera_type, img);
      // img = lidar2cam::utils::undist_image(image, K, D, camera_type);
       if(detect){
        cv::Mat drawImg, charucoCorners, charucoIds;
        bool bsuc = lidar2cam::utils::detectCharucoCorners(boardSize, squareSize, markerSize, image, drawImg, charucoCorners, charucoIds);
        if(bsuc){
          cv::Mat undistortedPoints;
          lidar2cam::utils::undistPointsGeneral(charucoCorners, K, D, camera_type, undistortedPoints);
          cv::aruco::drawDetectedCornersCharuco(img, undistortedPoints, charucoIds, cv::Scalar(255, 0, 0));
        }
       }
    }
    else if("dist" == action_modal){

    }
    boost::filesystem::path filepath(imageFilenames[i]);
    std::string out_file = outputDir + "/" + filepath.filename().string();
    cv::imwrite(out_file, img);
  }

  return 0;
}