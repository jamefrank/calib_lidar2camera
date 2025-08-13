#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Core>



struct Target2ImgReprojectionError
{
    Target2ImgReprojectionError(
        const Eigen::Vector3d& obj, 
        const Eigen::Vector2d& corner, 
        const Eigen::Matrix3d& K
        )
        : point_target_(obj), corner_cam2_(corner), K_(K) {}

    // target2cam rvec+tvec
    template <typename T>
    bool operator()(const T* const target2cam, const T* const fxy, T* residuals) const{
        //
        T p_target[3];
        p_target[0] = T(point_target_(0));
        p_target[1] = T(point_target_(1));
        p_target[2] = T(point_target_(2));
        // 
        T p_cam[3];
        ceres::AngleAxisRotatePoint(target2cam, p_target, p_cam);
        p_cam[0] += target2cam[3];
        p_cam[1] += target2cam[4];
        p_cam[2] += target2cam[5];

        //
        T u = p_cam[0] / p_cam[2];
        T v = p_cam[1] / p_cam[2];

        T fx = T(K_(0,0));
        T cx = T(K_(0,2));
        T fy = T(K_(1,1));
        T cy = T(K_(1,2));

        T p_x_1 = fxy[0]*u + cx;
        T p_y_1 = fxy[1]*v + cy;

        T p_x_2 = (T(corner_cam2_(0))-cx)/fx*fxy[0] + cx;
        T p_y_2 = (T(corner_cam2_(1))-cy)/fy*fxy[1] + cy;
        
        residuals[0] = p_x_1 - p_x_2;
        residuals[1] = p_y_1 - p_y_2;

        return true;
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3d& obj, 
        const Eigen::Vector2d& corner, 
        const Eigen::Matrix3d& K)
    {
        return (new ceres::AutoDiffCostFunction<Target2ImgReprojectionError, 2, 6, 2>(new Target2ImgReprojectionError(obj, corner, K)));
    }

    // members
    Eigen::Vector3d point_target_;
    Eigen::Vector2d corner_cam2_;
    Eigen::Matrix3d K_;
};


