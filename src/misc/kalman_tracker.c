#include "kalman_tracker.h"

KalmanFilterXYAH::KalmanFilterXYAH(float std_weight_pos,
                                   float std_weight_vel)
    : std_w_pos_(std_weight_pos),
      std_w_vel_(std_weight_vel),
      observation_mat_(Matrix4x8f::Zero())
{
    observation_mat_.template block<4,4>(0,0) = Matrix4f::Identity();
}

void KalmanFilterXYAH::initiate(const Vector4f& measurement,
                                Vector8f& mean,
                                Matrix8f& covariance) const
{
    mean.head<4>() = measurement;
    mean.tail<4>().setZero();

    float h = measurement(3);
    std::array<float,8> stds = {
        2 * std_w_pos_ * h,
        2 * std_w_pos_ * h,
        1e-2f,
        2 * std_w_pos_ * h,
        10 * std_w_vel_ * h,
        10 * std_w_vel_ * h,
        1e-5f,
        10 * std_w_vel_ * h
    };

    covariance.setZero();
    for (int i = 0; i < 8; ++i) {
        covariance(i,i) = stds[i] * stds[i];
    }
}

void KalmanFilterXYAH::predict(const Vector8f& mean,
                               const Matrix8f& cov,
                               float dt,
                               Vector8f& pred_mean,
                               Matrix8f& pred_covariance) const
{
    // State transition
    Matrix8f F = Matrix8f::Identity();
    for (int i = 0; i < 4; ++i) {
        F(i, i + 4) = dt;
    }

    float h = std::abs(mean(3));
    std::array<float,8> stds;
    stds.fill(std_w_pos_ * h);
    for (int i = 4; i < 8; ++i) {
        stds[i] = std_w_vel_ * h;
    }
    stds[2] = 1e-2f;
    stds[6] = 1e-5f;

    Matrix8f Q = Matrix8f::Zero();
    for (int i = 0; i < 8; ++i) {
        Q(i,i) = stds[i] * stds[i];
    }

    pred_mean       = F * mean;
    pred_covariance = F * cov * F.transpose() + Q;
}

void KalmanFilterXYAH::project(const Vector8f& mean,
                               const Matrix8f& cov,
                               Vector4f& proj_mean,
                               Matrix4f& proj_covariance) const
{
    proj_mean = observation_mat_ * mean;

    float h = std::abs(mean(3));
    std::array<float,4> stds = { std_w_pos_*h, std_w_pos_*h, 1e-1f, std_w_pos_*h };

    proj_covariance = observation_mat_ * cov * observation_mat_.transpose();
    for (int i = 0; i < 4; ++i) {
        proj_covariance(i,i) += stds[i] * stds[i];
    }
}

void KalmanFilterXYAH::update(const Vector8f& mean,
                              const Matrix8f& cov,
                              const Vector4f& measurement,
                              Vector8f& updated_mean,
                              Matrix8f& updated_covariance) const
{
    Vector4f proj_mean;
    Matrix4f proj_cov;
    project(mean, cov, proj_mean, proj_cov);

    // Kalman gain
    Matrix8x4f PHt = cov * observation_mat_.transpose();
    Eigen::LDLT<Matrix4f> solver(proj_cov);
    Matrix4f inv_proj = solver.solve(Matrix4f::Identity());
    Matrix8x4f K     = PHt * inv_proj;

    Vector4f innovation = measurement - proj_mean;
    updated_mean        = mean + K * innovation;
    updated_covariance = cov - K * proj_cov * K.transpose();
}

KalmanBoxTracker::KalmanBoxTracker(const Vector4f& init_bbox,
                                   double init_time)
    : kf_(), mean_(), covariance_(), last_update_time_(init_time)
{
    Vector4f meas = to_cxcyah(init_bbox);
    kf_.initiate(meas, mean_, covariance_);
}

Vector4f KalmanBoxTracker::predict(double predict_time)
{
    float dt = static_cast<float>((predict_time - last_update_time_) * 7.5);
    kf_.predict(mean_, covariance_, dt, mean_, covariance_);

    float cx = mean_(0);
    float cy = mean_(1);
    float a  = mean_(2);
    float h  = mean_(3);
    float w  = a * h;

    return Vector4f{ cx - w/2.f, cy - h/2.f, cx + w/2.f, cy + h/2.f };
}

void KalmanBoxTracker::update(const Vector4f& bbox, double curr_time)
{
    float dt = static_cast<float>((curr_time - last_update_time_) * 7.5);
    Vector4f meas = to_cxcyah(bbox);

    Vector8f pred_mean;
    Matrix8f pred_cov;
    kf_.predict(mean_, covariance_, dt, pred_mean, pred_cov);
    kf_.update(pred_mean, pred_cov, meas, mean_, covariance_);

    last_update_time_ = curr_time;
}

Vector4f KalmanBoxTracker::to_cxcyah(const Vector4f& xyxy)
{
    float w  = xyxy(2) - xyxy(0);
    float h  = xyxy(3) - xyxy(1);
    float cx = 0.5f * (xyxy(0) + xyxy(2));
    float cy = 0.5f * (xyxy(1) + xyxy(3));

    return Vector4f{ cx, cy, w / h, h };
}
