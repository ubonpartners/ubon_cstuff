
#include <Eigen/Dense>
#include <vector>

using Vector8f = Eigen::Matrix<float, 8, 1>;
using Matrix8f = Eigen::Matrix<float, 8, 8>;
using Vector4f = Eigen::Matrix<float, 4, 1>;
using Matrix4f = Eigen::Matrix<float, 4, 4>;
using Matrix4x8f = Eigen::Matrix<float, 4, 8>;
using Matrix8x4f = Eigen::Matrix<float, 8, 4>;

class KalmanFilterXYAH {
public:
    KalmanFilterXYAH(float std_weight_position = 1.0f/20.0f,
                     float std_weight_velocity = 1.0f/160.0f)
        : std_w_pos(std_weight_position), std_w_vel(std_weight_velocity)
    {
        // Observation model: selects [x,y,a,h] from the 8D state
        update_mat = Matrix4x8f::Zero();
        update_mat.block<4,4>(0,0) = Matrix4f::Identity();
    }

    void initiate(const Vector4f& meas, Vector8f& mean, Matrix8f& cov) const {
        mean.head<4>() = meas;
        mean.tail<4>().setZero();

        float h = meas(3);
        std::array<float,8> stds = {
            2 * std_w_pos * h,
            2 * std_w_pos * h,
            1e-2f,
            2 * std_w_pos * h,
            10 * std_w_vel * h,
            10 * std_w_vel * h,
            1e-5f,
            10 * std_w_vel * h
        };
        cov = Matrix8f::Zero();
        for(int i = 0; i < 8; ++i) {
            cov(i,i) = stds[i] * stds[i];
        }
    }

    void predict(const Vector8f& mean, const Matrix8f& cov,
                 float dt, Vector8f& pm, Matrix8f& pC) const
    {
        Matrix8f F = Matrix8f::Identity();
        for(int i = 0; i < 4; ++i) F(i, i+4) = dt;

        float h = std::abs(mean(3));
        std::array<float,8> stds;
        for(int i = 0; i < 4; ++i) stds[i] = std_w_pos * h;
        stds[2] = 1e-2f;
        stds[6] = 1e-5f;
        for(int i = 4; i < 8; ++i) stds[i] = std_w_vel * h;

        Matrix8f Q = Matrix8f::Zero();
        for(int i = 0; i < 8; ++i) Q(i,i) = stds[i] * stds[i];

        pm = F * mean;
        pC = F * cov * F.transpose() + Q;
    }

    void project(const Vector8f& mean, const Matrix8f& cov,
                 Vector4f& projMean, Matrix4f& projCov) const
    {
        float h = std::abs(mean(3));
        std::array<float,4> stds = { std_w_pos*h, std_w_pos*h, 1e-1f, std_w_pos*h };
        Matrix4f R = Matrix4f::Zero();
        for(int i = 0; i < 4; ++i) R(i,i) = stds[i] * stds[i];

        projMean = update_mat * mean;
        projCov  = update_mat * cov * update_mat.transpose() + R;
    }

    void update(const Vector8f& mean, const Matrix8f& cov,
                const Vector4f& meas, Vector8f& outMean, Matrix8f& outCov) const
    {
        Vector4f projMean;
        Matrix4f projCov;
        project(mean, cov, projMean, projCov);

        // Kalman gain: K = P * H^T * inv(S)
        Matrix8x4f PHt = cov * update_mat.transpose();
        Eigen::LDLT<Matrix4f> ldlt(projCov);
        Matrix4f invProj = ldlt.solve(Matrix4f::Identity());
        Eigen::Matrix<float,8,4> K = PHt * invProj;

        Vector4f innovation = meas - projMean;
        outMean = mean + K * innovation;
        outCov = cov - K * projCov * K.transpose();
    }

private:
    float std_w_pos;
    float std_w_vel;
    Matrix4x8f update_mat;
};

class KalmanBoxTracker {
public:
    KalmanBoxTracker(const Vector4f& init_xyxy, float init_time)
        : time(init_time)
    {
        Vector4f meas = xyxy_to_cxcyha(init_xyxy);
        kf.initiate(meas, mean, covariance);
    }

    Vector4f predict(float current_time) {
        float dt = (current_time - time) * 7.5f;
        kf.predict(mean, covariance, dt, mean, covariance);
        time = current_time;
        float cx = mean(0), cy = mean(1), a = mean(2), h = mean(3);
        float w = a * h;
        Vector4f xyxy;
        xyxy << cx - 0.5f * w,
                cy - 0.5f * h,
                cx + 0.5f * w,
                cy + 0.5f * h;
        return xyxy;
    }

    void update(const Vector4f& new_xyxy, float current_time) {
        float dt = (current_time - time) * 7.5f;
        Vector4f meas = xyxy_to_cxcyha(new_xyxy);
        Vector8f predMean;
        Matrix8f predCov;
        kf.predict(mean, covariance, dt, predMean, predCov);
        kf.update(predMean, predCov, meas, mean, covariance);
        time = current_time;
    }

private:
    static Vector4f xyxy_to_cxcyha(const Vector4f& xyxy) {
        float w = xyxy(2) - xyxy(0);
        float h = xyxy(3) - xyxy(1);
        float cx = 0.5f * (xyxy(0) + xyxy(2));
        float cy = 0.5f * (xyxy(1) + xyxy(3));
        Vector4f v;
        v << cx, cy, w/h, h;
        return v;
    }

    KalmanFilterXYAH kf;
    Vector8f mean;
    Matrix8f covariance;
    float time;
};