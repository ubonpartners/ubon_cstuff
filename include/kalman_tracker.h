// KalmanTracker.h
#ifndef KALMAN_TRACKER_H
#define KALMAN_TRACKER_H

#include <Eigen/Dense>
#include <array>
#include <cmath>

using Vector8f   = Eigen::Matrix<float, 8, 1>;
using Matrix8f   = Eigen::Matrix<float, 8, 8>;
using Vector4f   = Eigen::Matrix<float, 4, 1>;
using Matrix4f   = Eigen::Matrix<float, 4, 4>;
using Matrix4x8f = Eigen::Matrix<float, 4, 8>;
using Matrix8x4f = Eigen::Matrix<float, 8, 4>;

/**
 * @brief Constant velocity Kalman filter for [x, y, aspect ratio, height] state.
 */
class KalmanFilterXYAH {
public:
    KalmanFilterXYAH(float std_weight_pos = 1.f / 20.f,
                     float std_weight_vel = 1.f / 160.f);

    void initiate(const Vector4f& measurement,
                  Vector8f& mean, Matrix8f& covariance) const;

    void predict(const Vector8f& mean, const Matrix8f& covariance,
                 float dt, Vector8f& pred_mean,
                 Matrix8f& pred_covariance) const;

    void project(const Vector8f& mean, const Matrix8f& covariance,
                 Vector4f& proj_mean,
                 Matrix4f& proj_covariance) const;

    void update(const Vector8f& mean, const Matrix8f& covariance,
                const Vector4f& measurement,
                Vector8f& updated_mean,
                Matrix8f& updated_covariance) const;

private:
    float std_w_pos_;
    float std_w_vel_;
    Matrix4x8f observation_mat_;
};

/**
 * @brief Tracks 2D bounding boxes using a Kalman filter.
 */
class KalmanBoxTracker {
public:
    explicit KalmanBoxTracker(const Vector4f& init_bbox,
                              double init_time = 0.0);

    Vector4f predict(double predict_time);
    void     update(const Vector4f& bbox, double curr_time);

private:
    static Vector4f to_cxcyah(const Vector4f& xyxy);

    KalmanFilterXYAH kf_;
    Vector8f         mean_;
    Matrix8f         covariance_;
    double           last_update_time_;
};

#endif // KALMAN_TRACKER_H