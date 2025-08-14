#include <eigen3/Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>


namespace lidar2cam{
    enum class CameraType {
        PINHOLE,
        FISHEYE
    };

    namespace utils{
        bool parse_K(const YAML::Node& config, std::string name, cv::Mat& K);
        void parse_D(const YAML::Node& config, std::string name, cv::Mat& D);
        void log_cvmat(const cv::Mat& mat, const std::string& name);

        cv::Mat undist_image(const cv::Mat& src, const cv::Mat& K, const cv::Mat& D, CameraType type);
    
        std::vector<std::string> get_file_lists(const std::string& input_dir, const std::string& prefix, const std::string& fileExtension);
        cv::Mat genObjs(const cv::Size& boardSize, float squareSize);

        bool detectCharucoCorners(const cv::Size& boardSize, float squareSize, float markerSize, const cv::Mat& img, cv::Mat& draw_img, cv::Mat& charucoCorners, cv::Mat& charucoIds);
        bool estimateUndistPose(const cv::Size& boardSize, float squareSize, float markerSize, const cv::Mat& objs, const cv::Mat& K, const cv::Mat& charucoCorners, const cv::Mat& charucoIds, cv::Mat& rvec, cv::Mat& tvec, cv::Mat& draw_img, double& rerror);

        double reproject_error(const cv::Mat& corners, const cv::Mat& objs, const cv::Mat& K, const cv::Mat& rvec, const cv::Mat& tvec);

        std::vector<cv::Point2f> load_named_img_points(const std::string& named_img_file);
        std::vector<cv::Point3f> load_named_pcd_points(const std::string& named_pcd_file);

        void undistPointsGeneral(const cv::Mat& distortedPoints, const cv::Mat& K, const cv::Mat& D, CameraType type, cv::Mat& undistortedPoints);
        void undistImageGeneral(const cv::Mat& distortedImg, const cv::Mat& K, const cv::Mat& D, CameraType type, cv::Mat& undistortedImg);
        void distPointsGeneral(const cv::Mat& undistortedPoints, const cv::Mat& K, const cv::Mat& D, CameraType type, cv::Mat& distortedPoints);
        void distImageGeneral(const cv::Mat& undistortedImg, const cv::Mat& K, const cv::Mat& D, CameraType type, cv::Mat& distortedImg);

        void drawPoints(cv::Mat &img, const cv::Mat& points);
        void savePoints(const cv::Mat& points, const std::string& fileName);

    }
}