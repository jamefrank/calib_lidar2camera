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
  std::vector<cv::Mat> all_corner_ids;

  for (int i = 0; i < imageFilenames.size(); i++) {
    cv::Mat image = cv::imread(imageFilenames[i], -1); // distort
    
    cv::Mat imageDraw_dist, charucoCorners_dist, charucoIds_dist;
    bool bsuc1 = lidar2cam::utils::detectCharucoCorners(boardSize, squareSize, markerSize, image, imageDraw_dist, charucoCorners_dist, charucoIds_dist);

    cv::Mat undistImage = lidar2cam::utils::undist_image(image, K, D, camera_type);
    cv::Mat imageDraw_undist, charucoCorners_undist, charucoIds_undist;
    bool bsuc2 = lidar2cam::utils::detectCharucoCorners(boardSize, squareSize, markerSize, undistImage, imageDraw_undist, charucoCorners_undist, charucoIds_undist);

    double rerror = 0;
    cv::Mat rvec,tvec;
    bool bsuc3 = lidar2cam::utils::estimateUndistPose(boardSize, squareSize, markerSize, objs, K, charucoCorners_undist, charucoIds_undist, rvec, tvec, imageDraw_undist, rerror);

    if (bsuc1 && bsuc2 && bsuc3) {
      boost::filesystem::path filepath(imageFilenames[i]);
      spdlog::info("{} charuco corners rerror: {:.2f}",
                   filepath.filename().string(), rerror);

      all_corners.emplace_back(charucoCorners_dist);
      all_rvecs.emplace_back(rvec);
      all_tvecs.emplace_back(tvec);
      all_corner_ids.emplace_back(charucoIds_dist);

      if (verbose) {
        std::string out_file = outputDir + "/" + filepath.filename().string();
        cv::imwrite(out_file, imageDraw_undist);
      }
    }
  }
  spdlog::info("total charuco imgs: {} ", all_corners.size());
  assert(all_corners.size() >= 3);

  // load named points
  std::string named_img_file = inputDir + "/named_img.txt";
  std::string named_pcd_file = inputDir + "/named_pcd.txt";
  std::vector<cv::Point2f> named_img_points =
      lidar2cam::utils::load_named_img_points(named_img_file);
  std::vector<cv::Point3f> named_pcd_points =
      lidar2cam::utils::load_named_pcd_points(named_pcd_file);
  assert(named_img_points.size() == named_pcd_points.size());
  assert(named_img_points.size() >= 4);

//   // init lidar2cam RT
//   cv::Mat l2c_rvec, l2c_tvec;
//   if (!cv::solvePnP(named_pcd_points, named_img_points, K, cv::Mat(), l2c_rvec,
//                     l2c_tvec)) {
//     spdlog::error("slove pnp for init RT error");
//     exit(-1);
//   }
//   double init_error = lidar2cam::utils::reproject_error(cv::Mat(named_img_points), cv::Mat(named_pcd_points), K, l2c_rvec, l2c_tvec);
//   spdlog::info("init error: {}", init_error);
//   for(int i=0; i<all_corners.size(); i++){
//     double tmp_error = lidar2cam::utils::reproject_error(all_corners[i], objs, K, all_rvecs[i], all_tvecs[i]);
//     spdlog::info("init error {}: {}", i, tmp_error);
//   }
//   cv::Mat init_R;
//   cv::Rodrigues(l2c_rvec, init_R);
//   lidar2cam::utils::log_cvmat(init_R, "init_R");
//   lidar2cam::utils::log_cvmat(l2c_tvec, "init_t");
//   lidar2cam::utils::log_cvmat(K, "init_K");

//   // opt fx,fy and lidar2cam RT
//   spdlog::info("start opt by ceres ...");
//   double *parameters_ = new double[6 * (all_corners.size())];
//   for (int i = 0; i < all_corners.size(); i++) {
//     parameters_[i * 6 + 0] = all_rvecs[i].at<double>(0);
//     parameters_[i * 6 + 1] = all_rvecs[i].at<double>(1);
//     parameters_[i * 6 + 2] = all_rvecs[i].at<double>(2);
//     parameters_[i * 6 + 3] = all_tvecs[i].at<double>(0);
//     parameters_[i * 6 + 4] = all_tvecs[i].at<double>(1);
//     parameters_[i * 6 + 5] = all_tvecs[i].at<double>(2);
//     lidar2cam::utils::log_cvmat(all_rvecs[i], "rvec");
//     std::cout << parameters_[i * 6 + 0] << ":" << parameters_[i * 6 + 1] << ":" << parameters_[i * 6 + 2] << std::endl;
//     lidar2cam::utils::log_cvmat(all_tvecs[i], "tvec");
//     std::cout << parameters_[i * 6 + 3] << ":" << parameters_[i * 6 + 4] << ":" << parameters_[i * 6 + 5] << std::endl;

//   }
// //   parameters_[all_corners.size() * 6 + 0] = l2c_rvec.at<float>(0);
// //   parameters_[all_corners.size() * 6 + 1] = l2c_rvec.at<float>(1);
// //   parameters_[all_corners.size() * 6 + 2] = l2c_rvec.at<float>(2);
// //   parameters_[all_corners.size() * 6 + 3] = l2c_tvec.at<float>(0);
// //   parameters_[all_corners.size() * 6 + 4] = l2c_tvec.at<float>(1);
// //   parameters_[all_corners.size() * 6 + 5] = l2c_tvec.at<float>(2);
//   double parameters_l2c[6];
//     parameters_l2c[0] = l2c_rvec.at<float>(0);
//     parameters_l2c[1] = l2c_rvec.at<float>(1);
//     parameters_l2c[2] = l2c_rvec.at<float>(2);
//     parameters_l2c[3] = l2c_tvec.at<float>(0);
//     parameters_l2c[4] = l2c_tvec.at<float>(1);
//     parameters_l2c[5] = l2c_tvec.at<float>(2);

//   Eigen::Matrix3d K_Eigen;
//   cv::cv2eigen(K, K_Eigen);
//   double parameters_fxy_[2];
//   parameters_fxy_[0] = K_Eigen(0,0);
//   parameters_fxy_[1] = K_Eigen(1,1);
//   spdlog::info("fxy: {}, {}", parameters_fxy_[0], parameters_fxy_[1]);

//   ceres::Problem problem;

// //    lidar2cam::utils::log_cvmat(objs, "objs");
// //   add target board corner items
//   for (int i = 0; i < all_corners.size(); i++) {
//     for (int k = 0; k < objs.rows; k++) {
//         Eigen::Vector3d obj(objs.at<cv::Vec3f>(k)[0], objs.at<cv::Vec3f>(k)[1], objs.at<cv::Vec3f>(k)[2]);
//         Eigen::Vector2d corner(all_corners[i].at<cv::Vec2f>(k)[0], all_corners[i].at<cv::Vec2f>(k)[1]);
//         ceres::CostFunction *cost_funciton = Target2ImgReprojectionError::Create(obj, corner, K_Eigen);
//         ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
//         problem.AddResidualBlock(cost_funciton, lossFunction, parameters_ + i * 6, parameters_fxy_);
//     }
//     problem.SetParameterBlockConstant(parameters_ + i * 6);
//   }

// //   //add named points items
// //   for(int i=0; i<named_img_points.size(); i++){
// //     Eigen::Vector3d obj(named_pcd_points[i].x, named_pcd_points[i].y, named_pcd_points[i].z);
// //     Eigen::Vector2d corner(named_img_points[i].x, named_img_points[i].y);
// //     ceres::CostFunction *cost_funciton = Target2ImgReprojectionError::Create(obj, corner, K_Eigen);
// //     ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
// //     problem.AddResidualBlock(cost_funciton, lossFunction, parameters_l2c, parameters_fxy_);
// //   }

//   ceres::Solver::Options options;
//   options.linear_solver_type = ceres::DENSE_SCHUR;
//   options.minimizer_progress_to_stdout = false;
//   options.max_num_iterations = 200;
//   options.num_threads = 8;

//   ceres::Solver::Summary summary;
//   ceres::Solve(options, &problem, &summary);
//   std::cout << summary.FullReport() << "\n";
// //   std::cout << summary.BriefReport() << " \n";

//   l2c_rvec.at<float>(0) = parameters_l2c[0];
//   l2c_rvec.at<float>(1) = parameters_l2c[1];
//   l2c_rvec.at<float>(2) = parameters_l2c[2];
//   l2c_tvec.at<float>(0) = parameters_l2c[3];
//   l2c_tvec.at<float>(1) = parameters_l2c[4];
//   l2c_tvec.at<float>(2) = parameters_l2c[5];
//   K.at<float>(0,0) = parameters_fxy_[0];
//   K.at<float>(1,1) = parameters_fxy_[1];

//   double final_error = lidar2cam::utils::reproject_error(cv::Mat(named_img_points), cv::Mat(named_pcd_points), K, l2c_rvec, l2c_tvec);
//   spdlog::info("final error: {}", final_error);
//   for(int i=0; i<all_corners.size(); i++){
//     double tmp_error = lidar2cam::utils::reproject_error(all_corners[i], objs, K, all_rvecs[i], all_tvecs[i]);
//     spdlog::info("final error {}: {}", i, tmp_error);
//   }

//   cv::Mat final_R;
//   cv::Rodrigues(l2c_rvec, final_R);
//   lidar2cam::utils::log_cvmat(final_R, "lidar2cam_R");
//   lidar2cam::utils::log_cvmat(l2c_tvec, "lidar2cam_t");
//   lidar2cam::utils::log_cvmat(K, "final_K");
//   spdlog::info("fxy: {}, {}", parameters_fxy_[0], parameters_fxy_[1]);


//   delete [] parameters_;

  return 0;
}