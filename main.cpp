#include "ba_calib.h"
#include "cmdline.h"
#include "utils.h"
#include <Eigen/src/Core/Matrix.h>
#include <cassert>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/eigen.hpp>
#include <spdlog/spdlog.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>

#include <vector>
using namespace std;

int main(int argc, char *argv[]) {
  spdlog::info("Welcome to use calibLidar2cam(named point) calib tool!");

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
  parser.add<string>("output", 'o', "Output directory containing calib results",
                     true, "");
  parser.add("verbose", '\0', "verbose when calib");

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
  std::string outputDir = parser.get<std::string>("output");

  std::vector<cv::Mat> all_corners;
  std::vector<cv::Mat> all_rvecs;
  std::vector<cv::Mat> all_tvecs;

  for (int i = 0; i < imageFilenames.size(); i++) {
    cv::Mat image = cv::imread(imageFilenames[i], -1); // distort
    cv::Mat undistImage =
        lidar2cam::utils::undist_image(image, K, D, camera_type);
    double rerror = 0;
    cv::Mat imageCopy, corners, charucoIds, rvec, tvec;
    bool bsuc = lidar2cam::utils::detectCharucoCornersAndPose(
        boardSize, squareSize, markerSize, undistImage, K, objs, imageCopy,
        corners, charucoIds, rvec, tvec, rerror);

    if (bsuc) {
      boost::filesystem::path filepath(imageFilenames[i]);
      spdlog::info("{} charuco corners rerror: {:.2f}",
                   filepath.filename().string(), rerror);

      all_corners.emplace_back(corners);
      all_rvecs.emplace_back(rvec);
      all_tvecs.emplace_back(tvec);

      if (verbose) {
        std::string out_file = outputDir + "/" + filepath.filename().string();
        cv::imwrite(out_file, imageCopy);
      }
    }
  }
  spdlog::info("detect imgs: {} ", all_corners.size());
  assert(all_corners.size() >= 3);

  // load named points
  std::string named_img_file = inputDir + "/named_img.txt";
  std::string named_pcd_file = inputDir + "/named_pcd.txt";
  std::vector<cv::Point2d> named_img_points =
      lidar2cam::utils::load_named_img_points(named_img_file);
  std::vector<cv::Point3d> named_pcd_points =
      lidar2cam::utils::load_named_pcd_points(named_pcd_file);
  assert(named_img_points.size() == named_pcd_points.size());
  assert(named_img_points.size() >= 4);

  // init lidar2cam RT
  cv::Mat l2c_rvec, l2c_tvec;
  if (!cv::solvePnP(named_pcd_points, named_img_points, K, cv::Mat(), l2c_rvec,
                    l2c_tvec)) {
    spdlog::error("slove pnp for init RT error");
    exit(-1);
  }
  double init_error = lidar2cam::utils::reproject_error(cv::Mat(named_img_points), cv::Mat(named_pcd_points), K, l2c_rvec, l2c_tvec);
  spdlog::info("init error: {}", init_error);

  // opt fx,fy and lidar2cam RT
  spdlog::info("start opt by ceres ...");
  double *parameters_ = new double[6 * (all_corners.size() + 1)];
  for (int i = 0; i < all_corners.size(); i++) {
    parameters_[i * 6 + 0] = all_rvecs[i].at<double>(0);
    parameters_[i * 6 + 1] = all_rvecs[i].at<double>(1);
    parameters_[i * 6 + 2] = all_rvecs[i].at<double>(2);
    parameters_[i * 6 + 3] = all_tvecs[i].at<double>(0);
    parameters_[i * 6 + 4] = all_tvecs[i].at<double>(1);
    parameters_[i * 6 + 5] = all_tvecs[i].at<double>(2);
  }
  parameters_[all_corners.size() * 6 + 0] = l2c_rvec.at<double>(0);
  parameters_[all_corners.size() * 6 + 1] = l2c_rvec.at<double>(1);
  parameters_[all_corners.size() * 6 + 2] = l2c_rvec.at<double>(2);
  parameters_[all_corners.size() * 6 + 3] = l2c_tvec.at<double>(0);
  parameters_[all_corners.size() * 6 + 4] = l2c_tvec.at<double>(1);
  parameters_[all_corners.size() * 6 + 5] = l2c_tvec.at<double>(2);

  Eigen::Matrix3d K_Eigen;
  cv::cv2eigen(K, K_Eigen);
  double parameters_fxy_[2];
  parameters_fxy_[0] = K_Eigen(0,0);
  parameters_fxy_[1] = K_Eigen(1,1);

  ceres::Problem problem;

  // add target board corner items
  for (int i = 0; i < all_corners.size(); i++) {
    for (int k = 0; k < objs.rows; k++) {
        Eigen::Vector3d obj(objs.at<cv::Vec3f>(k)[0], objs.at<cv::Vec3f>(k)[1], objs.at<cv::Vec3f>(k)[2]);
        Eigen::Vector2d corner(all_corners[i].at<cv::Vec2f>(k)[0], all_corners[i].at<cv::Vec2f>(k)[1]);
        ceres::CostFunction *cost_funciton = Target2ImgReprojectionError::Create(obj, corner, K_Eigen);
        ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
        problem.AddResidualBlock(cost_funciton, lossFunction, parameters_ + i * 6, parameters_fxy_);
    }
  }

  //add named points items
  for(int i=0; i<named_img_points.size(); i++){
    Eigen::Vector3d obj(named_pcd_points[i].x, named_pcd_points[i].y, named_pcd_points[i].z);
    Eigen::Vector2d corner(named_img_points[i].x, named_img_points[i].y);
    ceres::CostFunction *cost_funciton = Target2ImgReprojectionError::Create(obj, corner, K_Eigen);
    ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
    problem.AddResidualBlock(cost_funciton, lossFunction, parameters_ + all_corners.size() * 6, parameters_fxy_);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 200;
  options.num_threads = 8;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  l2c_rvec.at<double>(0) = parameters_[6*all_corners.size() + 0];
  l2c_rvec.at<double>(1) = parameters_[6*all_corners.size() + 1];
  l2c_rvec.at<double>(2) = parameters_[6*all_corners.size() + 2];
  l2c_tvec.at<double>(0) = parameters_[6*all_corners.size() + 3];
  l2c_tvec.at<double>(1) = parameters_[6*all_corners.size() + 4];
  l2c_tvec.at<double>(2) = parameters_[6*all_corners.size() + 5];
  K.at<double>(0,0) = parameters_fxy_[0];
  K.at<double>(1,1) = parameters_fxy_[1];

  double final_error = lidar2cam::utils::reproject_error(cv::Mat(named_img_points), cv::Mat(named_pcd_points), K, l2c_rvec, l2c_tvec);
  spdlog::info("final error: {}", final_error);

  cv::Mat final_R;
  cv::Rodrigues(l2c_rvec, final_R);
  lidar2cam::utils::log_cvmat(final_R, "lidar2cam_R");
  lidar2cam::utils::log_cvmat(l2c_tvec, "lidar2cam_t");

  return 0;
}