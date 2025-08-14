#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Core>



struct FishEyeReprojectionError
{
    FishEyeReprojectionError(
        const Eigen::Vector3d& obj, 
        const Eigen::Vector2d& corner, 
        double cx, double cy,
        double k2, double k3, double k4, double k5,
        double weight
        )
        : point_target_(obj), corner_cam2_(corner), cx_(cx), cy_(cy), k2_(k2), k3_(k3), k4_(k4), k5_(k5), weight_(weight) {}

    // target2cam rvec+tvec
    template <typename T>
    bool operator()(const T* const target2cam, const T* const fxy, T* residuals) const{
        // world 2 cam
        T p_target[3];
        p_target[0] = T(point_target_(0));
        p_target[1] = T(point_target_(1));
        p_target[2] = T(point_target_(2));
        
        T P_c[3];
        ceres::AngleAxisRotatePoint(target2cam, p_target, P_c);
        P_c[0] += target2cam[3];
        P_c[1] += target2cam[4];
        P_c[2] += target2cam[5];

        // project 3D object point to the image plane;
        T k2 = T(k2_);
        T k3 = T(k3_);
        T k4 = T(k4_);
        T k5 = T(k5_);
        T mu = fxy[0];
        T mv = fxy[1];
        T u0 = T(cx_);
        T v0 = T(cy_);

        T len = sqrt(P_c[0] * P_c[0] + P_c[1] * P_c[1] + P_c[2] * P_c[2]);
        T theta = acos(P_c[2] / len);
        T phi = atan2(P_c[1], P_c[0]);

        Eigen::Matrix<T, 2, 1> p_u = r(k2, k3, k4, k5, theta) * Eigen::Matrix<T, 2, 1>(cos(phi), sin(phi));

        T xd = mu * p_u(0) + u0;
        T yd = mv * p_u(1) + v0;

        residuals[0] = T(weight_)*(xd - T(corner_cam2_(0)));
        residuals[1] = T(weight_)*(yd - T(corner_cam2_(1)));

        return true;
    }

    template <typename T>
    T r(T k2, T k3, T k4, T k5, T theta) const {
        // k1 = 1
        return theta + k2 * theta * theta * theta + k3 * theta * theta * theta * theta * theta + k4 * theta * theta * theta * theta * theta * theta * theta +
            k5 * theta * theta * theta * theta * theta * theta * theta * theta * theta;
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3d& obj, 
        const Eigen::Vector2d& corner, 
        double cx, double cy,
        double k2, double k3, double k4, double k5,
        double weight)
    {
        return (new ceres::AutoDiffCostFunction<FishEyeReprojectionError, 2, 6, 2>(new FishEyeReprojectionError(obj, corner, cx, cy, k2, k3, k4, k5, weight)));
    }

    // members
    Eigen::Vector3d point_target_;
    Eigen::Vector2d corner_cam2_;
    double cx_;
    double cy_;
    double k2_;
    double k3_;
    double k4_; 
    double k5_;
    double weight_;
};



struct PinholeReprojectionError
{
    PinholeReprojectionError(
        const Eigen::Vector3d& obj, 
        const Eigen::Vector2d& corner, 
        double cx, double cy,
        double k1, double k2, double k3, double p1, double p2,
        double weight
        )
        : point_target_(obj), corner_cam2_(corner), cx_(cx), cy_(cy), k1_(k1), k2_(k2), k3_(k3), p1_(p1), p2_(p2), weight_(weight) {}

    // target2cam rvec+tvec
    template <typename T>
    bool operator()(const T* const target2cam, const T* const fxy, T* residuals) const{
        // world 2 cam
        T p_target[3];
        p_target[0] = T(point_target_(0));
        p_target[1] = T(point_target_(1));
        p_target[2] = T(point_target_(2));
        
        T P_c[3];
        ceres::AngleAxisRotatePoint(target2cam, p_target, P_c);
        P_c[0] += target2cam[3];
        P_c[1] += target2cam[4];
        P_c[2] += target2cam[5];

        // project 3D object point to the image plane;
        T k1 = T(k1_);
        T k2 = T(k2_);
        T k3 = T(k3_);
        T p1 = T(p1_);
        T p2 = T(p2_);
        T alpha = T(0);  // cameraParams.alpha();
        T fx = fxy[0];
        T fy = fxy[1];
        T cx = T(cx_);
        T cy = T(cy_);

        T u = P_c[0] / P_c[2];
        T v = P_c[1] / P_c[2];

        T rho_sqr = u * u + v * v;
        T L = T(1.0) + k1 * rho_sqr + k2 * rho_sqr * rho_sqr;
        T du = T(2.0) * p1 * u * v + p2 * (rho_sqr + T(2.0) * u * u);
        T dv = p1 * (rho_sqr + T(2.0) * v * v) + T(2.0) * p2 * u * v;

        u = L * u + du;
        v = L * v + dv;
        T xd = fx * (u + alpha * v) + cx;
        T yd = fy * v + cy;

        residuals[0] = T(weight_)*(xd - T(corner_cam2_(0)));
        residuals[1] = T(weight_)*(yd - T(corner_cam2_(1)));

        return true;
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3d& obj, 
        const Eigen::Vector2d& corner, 
        double cx, double cy,
        double k1, double k2, double k3, double p1, double p2,
        double weight)
    {
        return (new ceres::AutoDiffCostFunction<PinholeReprojectionError, 2, 6, 2>(new PinholeReprojectionError(obj, corner, cx, cy, k1, k2, k3, p1, p2, weight)));
    }

    // members
    Eigen::Vector3d point_target_;
    Eigen::Vector2d corner_cam2_;
    double cx_;
    double cy_;
    double k1_;
    double k2_;
    double k3_;
    double p1_; 
    double p2_;
    double weight_;
};
