#include "utils.h"
#include <Eigen/src/Core/Matrix.h>
#include <cassert>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <vector>

bool lidar2cam::utils::parse_K(const YAML::Node &config, std::string name,
                               cv::Mat &K) {
  if (config[name]) {
    if (config[name].IsSequence()) {
      int num = config[name].size();
      if (9 == num || 3 == num) {
        std::vector<float> tmp;
        if (9 == num) {
          for (int i = 0; i < num; i++) {
            float value = config[name][i].as<float>();
            tmp.push_back(value);
          }
        }
        if (3 == num) {
          for (int i = 0; i < num; i++) {
            for (int j = 0; j < config[name][i].size(); ++j) {
              float value = config[name][i][j].as<float>();
              tmp.push_back(value);
            }
          }
        }
        K = cv::Mat(3, 3, CV_32F, tmp.data()).clone();
        return true;
      } else
        spdlog::error("3x3 or 9 supported");
    } else
      spdlog::error("[{0}] not list", name);
  } else {
    spdlog::error("{0} not exists", name);
  }

  return false;
}

void lidar2cam::utils::parse_D(const YAML::Node &config, std::string name,
                               cv::Mat &D_) {
  std::vector<float> D; // TODO
  for (std::size_t i = 0; i < config["D"].size(); ++i) {
    float value = config["D"][i].as<float>();
    D.push_back(value);
  }
  D_ = cv::Mat(1, D.size(), CV_32F, D.data()).clone();
}

void lidar2cam::utils::log_cvmat(const cv::Mat &mat, const std::string &name) {
  std::ostringstream oss;
  oss << name << " = " << std::endl << mat << std::endl;
  spdlog::info("{}", oss.str());
}

cv::Mat lidar2cam::utils::undist_image(const cv::Mat &src, const cv::Mat &K,
                                       const cv::Mat &D, CameraType type) {
  cv::Mat dst;
  if (type == CameraType::PINHOLE) {
    cv::undistort(src, dst, K, D);
  } else if (type == CameraType::FISHEYE) {
    cv::Mat map1;
    cv::Mat map2;
    cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat(), K, src.size(),
                                         CV_16SC2, map1, map2);
    cv::remap(src, dst, map1, map2, cv::INTER_AREA, cv::BORDER_CONSTANT);
  } else {
    spdlog::error("unknown camera type");
    exit(-1);
  }

  return dst;
}

std::vector<std::string>
lidar2cam::utils::get_file_lists(const std::string &inputDir,
                                 const std::string &prefix,
                                 const std::string &fileExtension) {
  std::vector<std::string> imageFilenames;
  //
  if (!boost::filesystem::exists(inputDir) &&
      !boost::filesystem::is_directory(inputDir)) {
    spdlog::error("Cannot find input directory{}", inputDir);
  }
  //
  boost::filesystem::directory_iterator itr;
  for (boost::filesystem::directory_iterator itr(inputDir);
       itr != boost::filesystem::directory_iterator(); ++itr) {
    if (!boost::filesystem::is_regular_file(itr->status())) {
      continue;
    }
    std::string filename = itr->path().filename().string();
    // check if prefix matches
    if (!prefix.empty()) {
      if (filename.compare(0, prefix.length(), prefix) != 0) {
        continue;
      }
    }
    // check if file extension matches
    if (filename.compare(filename.length() - fileExtension.length(),
                         fileExtension.length(), fileExtension) != 0) {
      continue;
    }

    imageFilenames.push_back(itr->path().string());
  }

  std::sort(imageFilenames.begin(), imageFilenames.end());

  return imageFilenames;
}

cv::Mat lidar2cam::utils::genObjs(const cv::Size &boardSize, float squareSize) {
  float squareLength = squareSize / 1000.0f;
  cv::Mat objs(boardSize.width * boardSize.height, 1, CV_32FC3);
  for (int i = 0; i < objs.rows; i++) {
    int idx = i % boardSize.width;
    int idy = i / boardSize.width;
    objs.at<cv::Vec3f>(i)[0] = (idx + 1) * squareLength;
    objs.at<cv::Vec3f>(i)[1] = (idy + 1) * squareLength;
    objs.at<cv::Vec3f>(i)[2] = 0;
  }
  return objs;
}

bool lidar2cam::utils::detectCharucoCornersAndPose(
    const cv::Size &boardSize, float squareSize, float markerSize,
    const cv::Mat &undistImg, const cv::Mat &K, const cv::Mat &objs,
    cv::Mat &imageCopy, cv::Mat &charucoCorners, cv::Mat &charucoIds,
    cv::Mat &rvec, cv::Mat &tvec, double &rerror) {
  int squaresX = boardSize.width + 1;
  int squaresY = boardSize.height + 1;
  float squareLength = squareSize / 1000.0f;
  float markerLength = markerSize / 1000.0f;
  cv::Ptr<cv::aruco::Dictionary> dictionary_ =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
  cv::Ptr<cv::aruco::CharucoBoard> board_ = cv::aruco::CharucoBoard::create(
      squaresX, squaresY, squareLength, markerLength, dictionary_);

  cv::Mat gray;
  undistImg.copyTo(imageCopy);
  cv::cvtColor(undistImg, gray, cv::COLOR_BGR2GRAY);
  cv::equalizeHist(gray, gray);
  // cv::Mat blur_usm;
  // cv::GaussianBlur(gray, blur_usm, cv::Size(0, 0), 25);
  // cv::addWeighted(gray, 1.5, blur_usm, -0.5, 0, gray);

  cv::Ptr<cv::aruco::DetectorParameters> parameters =
      cv::aruco::DetectorParameters::create();
  std::vector<int> marker_ids;
  std::vector<std::vector<cv::Point2f>> marker_corners, marker_rejected;

  cv::aruco::detectMarkers(gray, dictionary_, marker_corners, marker_ids,
                           parameters, marker_rejected);

  cv::aruco::interpolateCornersCharuco(marker_corners, marker_ids, gray, board_,
                                       charucoCorners, charucoIds);
  if (charucoIds.rows == boardSize.width * boardSize.height) {

    bool valid = cv::aruco::estimatePoseCharucoBoard(
        charucoCorners, charucoIds, board_, K, cv::Mat(), rvec, tvec);
    if (valid) {
      // calc error
      rerror = reproject_error(charucoCorners, objs, K, rvec, tvec);
      //
      cv::aruco::drawDetectedMarkers(imageCopy, marker_corners, marker_ids);
      cv::aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners,
                                            charucoIds, cv::Scalar(255, 0, 0));
      cv::drawFrameAxes(imageCopy, K, cv::Mat(), rvec, tvec, 0.1f);
      return true;
    }
  }

  return false;
}

double lidar2cam::utils::reproject_error(const cv::Mat &corners,
                                         const cv::Mat &objs, const cv::Mat &K,
                                         const cv::Mat &rvec,
                                         const cv::Mat &tvec) {
  cv::Mat imgpoints;
  cv::projectPoints(objs, rvec, tvec, K, cv::Mat(), imgpoints);
  double rerror = 0;
  for (int i = 0; i < imgpoints.rows; i++) {
    cv::Vec2f e = imgpoints.at<cv::Vec2f>(i) - corners.at<cv::Vec2f>(i);
    rerror += std::sqrt(e[0] * e[0] + e[1] * e[1]);
  }
  rerror /= imgpoints.rows;

  return rerror;
}

std::vector<cv::Point2d>
lidar2cam::utils::load_named_img_points(const std::string &named_img_file) {
  std::vector<cv::Point2d> points;

  std::ifstream file(named_img_file);
  assert(file.is_open());

  double x, y;
  while (file >> x >> y) {
    points.emplace_back(x, y);
  }

  file.close();

  return points;
}

std::vector<cv::Point3d>
lidar2cam::utils::load_named_pcd_points(const std::string &named_pcd_file) {
  std::vector<cv::Point3d> points;

  std::ifstream file(named_pcd_file);
  assert(file.is_open());

  double x, y, z;
  while (file >> x >> y >> z) {
    points.emplace_back(x, y, z);
  }

  file.close();

  return points;
}
