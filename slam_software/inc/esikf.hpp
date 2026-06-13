#include <nlohmann/json.hpp>
#include <mutex>
#include <condition_variable>
#include <pcl/kdtree/kdtree_flann.h>

#include "dataloader.hpp"
#include "so3_math.hpp"

class KF {
    using M3D = Eigen::Matrix3d;
    using V3D = Eigen::Vector3d;
    // Usage:
    // 1. init;
    // 2. forward... forward
    // 3. if get lidar, measurament update.
private:
    struct PlanarFeature {
        double residual;
        V3D normal;
        V3D p_loc;  // point local_frame
    
    };

    struct EdgeFeature {
        V3D residual;
        V3D direction;
        V3D p_loc;
    };
public:
    KF(nlohmann::json kf_cfg);
    bool init(const IMUMsg& msg);
    void forward_propagation(const IMUMsg& msg);
    void measurament_update(pcl::PointCloud<PointType>::Ptr cloud_glob, pcl::KdTreeFLANN<PointType>& kdtree, pcl::PointCloud<PointType>::Ptr frame_loc);
    // interface
    StatusManifold get_status();
    bool is_ready();
private:   
    void calc_Fx(const StatusTangent& t, const double dt, const V3D& acc);
    Eigen::Matrix<double, 18, 12> calc_FQ(const StatusTangent& t, const double dt);
    void calc_H_R(M3D& rotation, std::vector<PlanarFeature>& planar_buf, std::vector<EdgeFeature>& edge_buf);
    void calc_K_fastlio();
    Eigen::Matrix<double, 18, 18> calc_Jx();
    Eigen::MatrixXd diag_inv(Eigen::MatrixXd& m);

    void knn_residual_calc(pcl::PointCloud<PointType>::Ptr frame_glob, pcl::KdTreeFLANN<PointType>& kdtree, pcl::PointCloud<PointType>::Ptr global_map_match, const StatusManifold& status);
    Eigen::SelfAdjointEigenSolver<M3D> pca_feature(Eigen::Matrix<double, 3, 5>& m, PointType& p_mean);
    V3D edge_residual(const V3D& edge, const V3D& p_loc, const V3D& center);
    double planar_residual(const V3D& normal, const V3D& p_loc, const V3D& center);
    
public:
    std::mutex mtx_kf;
    std::condition_variable cv_kf;
    
private:
    std::vector<PlanarFeature> planar_buf;
    std::vector<EdgeFeature> edge_buf;
    
    StatusManifold last_status;
    StatusManifold prior_status;
    StatusManifold nomial_status;
    StatusTangent error_state;
    
    Eigen::Matrix<double, 12, 12> imu_bias_cov_Q;
    Eigen::Matrix<double, 18, 18> status_err_cov;
    // the ns timestamp of current prior status / nomial status
    // actually we the one of above once, never use them altogether.
    uint64_t timestamp = 0;
    
    Eigen::Matrix<double, 18, 18> F;
    Eigen::Matrix<double, 18, 12> FQ;
    Eigen::MatrixXd R;  // diagonal.
    M3D lidar_meas_cov_m3d;
    double lidar_meas_var;
    Eigen::VectorXd z;
    Eigen::MatrixXd H;
    Eigen::MatrixXd K;
    // init related
    bool ready = false;
    bool first_frame = true;
    int init_cnt = 0;
    int max_init_cnt = 0;
    V3D init_g = V3D::Zero();
    V3D init_bw = V3D::Zero();

    int max_iter = 3;
    double theta_norm = 0.0001;
    double pos_norm = 0.0001;
public:
    std::atomic<int> iter;
};

