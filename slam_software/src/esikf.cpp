#include "esikf.hpp"

KF::KF(const nlohmann::json kf_cfg) {
    nomial_status.R.setIdentity();
    nomial_status.ba.setZero();
    nomial_status.bw.setZero();
    nomial_status.g = V3D(0, 0, -9.81);
    nomial_status.p.setZero();
    nomial_status.v.setZero();
    
    last_status = nomial_status;  // set zero
    
    Eigen::Vector<double, 12> imu_bias;
    imu_bias.block(0, 0, 3, 1) = V3D::Constant(kf_cfg["imu_nw"]);
    imu_bias.block(3, 0, 3, 1) = V3D::Constant(kf_cfg["imu_na"]);
    imu_bias.block(6, 0, 3, 1) = V3D::Constant(kf_cfg["imu_nbw"]);
    imu_bias.block(9, 0, 3, 1) = V3D::Constant(kf_cfg["imu_nba"]);
    max_iter = kf_cfg["max_iter"];
    theta_norm = kf_cfg["theta_norm"];
    pos_norm = kf_cfg["pos_norm"];
    
    imu_bias_cov_Q = imu_bias.asDiagonal();
    lidar_meas_cov_m3d = V3D::Constant(kf_cfg["meas_xyz"]).asDiagonal();
    // status_err_cov = Eigen::Matrix<double, 18, 18>::Zero();
    status_err_cov = Eigen::Matrix<double, 18, 18>::Identity() * 1; 

    max_init_cnt = kf_cfg["init_imu_msgs"];

    F.setZero();
    FQ.setZero();
    
}

// you should keep the device stable stationary while init.
// return: the filter status after this init round.
bool KF::init(const IMUMsg& msg) {
    if (init_cnt == 0) {
        init_g.setZero();
        init_bw.setZero();
    }
    init_cnt += 1;

    init_g += msg.linear_acceleration;
    init_bw += msg.angular_velocity;

    if (init_cnt == max_init_cnt) {
        
        if (nomial_status.bw != V3D::Zero() || nomial_status.R != M3D::Identity()) {
            throw std::logic_error("init nomial first");
        }
        
        init_bw /= init_cnt;
        nomial_status.bw = init_bw;
        
        init_g /= init_cnt;
        Eigen::Quaterniond r_q = Eigen::Quaterniond::FromTwoVectors(init_g.normalized(), -nomial_status.g);
        nomial_status.R = r_q.toRotationMatrix();
        nomial_status.ba = init_g + nomial_status.R.transpose() * nomial_status.g; 
        
        ready = true;
    }
    return ready;
}

void KF::forward_propagation(const IMUMsg& msg) {
    if (ready != true) {
        throw std::runtime_error("init KF first.");
    }
    if (msg.timestamp <= timestamp) {
        throw std::runtime_error("imu msg earlier than current frame!");
    }

    if (first_frame) {
        timestamp = msg.timestamp;
        first_frame = false;
        return;
    }
    
    double dt = (double)(msg.timestamp - timestamp) / 1e9;
    timestamp = msg.timestamp;
    
    StatusTangent  increment;
    increment.dtheta = dt * (msg.angular_velocity - nomial_status.bw);
    increment.dp = dt * nomial_status.v;
    V3D est_a = nomial_status.R * (msg.linear_acceleration - nomial_status.ba);
    increment.dv = est_a * dt + nomial_status.g * dt;
    std::unique_lock<std::mutex> lock(mtx_kf);
    last_status = nomial_status;
    nomial_status = boxplus(nomial_status, increment);  // note that the signature of boxplus
    prior_status = nomial_status;
    lock.unlock();
    cv_kf.notify_all();

    // covariance update
    calc_Fx(increment, dt, msg.linear_acceleration - last_status.ba);
    calc_FQ(increment, dt);
    // P- = F P FT + F_Q Q F_Q T
    status_err_cov = F * status_err_cov * F.transpose()
                     + FQ * imu_bias_cov_Q * FQ.transpose();  // Cov = E[xx^T]
    
}


void KF::measurament_update(pcl::PointCloud<PointType>::Ptr cloud_glob, pcl::KdTreeFLANN<PointType>& kdtree, pcl::PointCloud<PointType>::Ptr frame_loc) {
    // solve MAP
    // planar_buf.clear();
    iter = 0;
    Eigen::Matrix<double, 18, 18> J;
    Eigen::Matrix<double, 18, 18> P;
    Eigen::Matrix<double, 18, 1> e;
    // pcl::PointCloud<PointType>::Ptr frame_glob (new pcl::PointCloud<PointType>);
    while (true) {
        iter ++;
        // frame_glob->clear();
        
        edge_buf.clear();
        planar_buf.clear();
        knn_residual_calc(frame_loc, kdtree, cloud_glob, nomial_status);
        
        calc_H_R(nomial_status.R, planar_buf, edge_buf);
        if (H.rows() == 0) {
            return;
        }
        // calc_K_fastlio();

        J = calc_Jx();
        P = J.inverse() * status_err_cov * J.inverse().transpose();
        K = (H.transpose() * diag_inv(R) * H + P.inverse()).inverse() * H.transpose() * diag_inv(R);
        e = -1. * K * z - (Eigen::Matrix<double, 18, 18>::Identity() - K * H) * P * J.inverse() * boxminus(nomial_status, prior_status);

        // std::unique_lock<std::mutex> lock(mtx_kf);
        nomial_status = boxplus(nomial_status, e);
        // lock.unlock();
        if ((e.block(0, 0, 3, 1).norm() < theta_norm && e.block(3, 0, 3, 1).norm() < pos_norm)
            || iter >= max_iter) {
            break;
        }
    }
    // posterior covariance update.
    // std::unique_lock<std::mutex> lock(mtx_kf);
    status_err_cov = (Eigen::Matrix<double, 18, 18>::Identity() - K * H) * P;
    // lock.unlock();
    cv_kf.notify_all();
    // // finish one solving.
}

StatusManifold KF::get_status() {
    std::lock_guard<std::mutex> lock(mtx_kf);
    return nomial_status;
}

bool KF::is_ready() {return ready;}
  
void KF::calc_Fx(const StatusTangent& t, const double dt, const V3D& acc) {
    M3D _R = Exp(t.dtheta);
    M3D J_r = SO3_right_jacobian(t.dtheta);
    F.setZero();
    F.block(0, 0, 3, 3) = _R.transpose();
    F.block(0, 9, 3, 3) = -1 * J_r * dt;
    F.block(3, 3, 3, 3).setIdentity();
    F.block(3, 6, 3, 3) = M3D::Identity() * dt;

    F.block(6, 0, 3, 3) = -1 * last_status.R * asymmetric(acc) * dt;
    F.block(6, 12, 3, 3) = -1 * last_status.R * dt;
    F.block(6, 15, 3, 3) = M3D::Identity() * dt;

    F.block(9, 9, 9, 9).setIdentity();
    
}

Eigen::Matrix<double, 18, 12> KF::calc_FQ(const StatusTangent& t, const double dt) {
    M3D J_r = SO3_right_jacobian(t.dtheta);
    FQ.setZero();
    FQ.block(0, 0, 3, 3) = J_r * dt;
    FQ.block(6, 3, 3, 3) = last_status.R * dt;
    FQ.block(9, 6, 3, 3) = M3D::Identity() * dt;
    FQ.block(12, 9, 3, 3) = M3D::Identity() * dt; 
    return FQ;
}

void KF::calc_H_R(M3D& rotation,
         std::vector<PlanarFeature>& planar_buf,
        std::vector<EdgeFeature>& edge_buf) {
    
    H.resize(planar_buf.size() + edge_buf.size() * 3, 18);
    z.resize(planar_buf.size() + edge_buf.size() * 3);
    R.resize(planar_buf.size() + edge_buf.size() * 3, planar_buf.size() + edge_buf.size() * 3);
    
    H.setZero();
    z.setZero();
    R.setZero();

    auto&& it_planar = planar_buf.begin();
    for (size_t i = 0; i < planar_buf.size(); i ++) {
        H.block(i, 0, 1, 3) = -1 * it_planar->normal.transpose() * rotation * asymmetric(it_planar->p_loc);
        H.block(i, 3, 1, 3) = it_planar->normal.transpose();
        z(i) = it_planar->residual;
        Eigen::Matrix<double, 1, 3> T = it_planar->normal.transpose() * rotation;            
        R(i, i) = T * lidar_meas_cov_m3d * T.transpose();
        it_planar ++;
    }
    
    auto&& it_edge = edge_buf.begin();
    for (size_t i = 0; i < edge_buf.size(); i++) {
        H.block(planar_buf.size() + i * 3, 0, 3, 3) = -1 * asymmetric(it_edge->direction) * rotation * asymmetric(it_edge->p_loc);
        H.block(planar_buf.size() + i * 3, 3, 3, 3) = asymmetric(it_edge->direction);
        z.block(planar_buf.size() + i * 3, 0, 3, 1) = it_edge->residual;
        M3D T = asymmetric(it_edge->direction) * rotation;
        R.block(planar_buf.size() + i * 3, planar_buf.size() + i * 3, 3, 3)
            = lidar_meas_cov_m3d;// T * lidar_meas_cov_m3d * T.transpose();
        it_edge ++;
    }
}

// (H^\topR^{-1}H + P^{-1})^{-1}H^\top R^{-1}
void KF::calc_K_fastlio() {
    K = -1 * (H.transpose() * diag_inv(R) * H + status_err_cov.inverse()).inverse() * H.transpose() * diag_inv(R);
    
}

Eigen::Matrix<double, 18, 18> KF::calc_Jx() {
    Eigen::Matrix<double, 18, 18> J;
    J.setIdentity();
    V3D main_part = Log(prior_status.R.transpose() * nomial_status.R);
    J.block(0, 0, 3, 3) = SO3_right_jacobian_inv(main_part);
    return J;
}

void KF::knn_residual_calc(pcl::PointCloud<PointType>::Ptr frame_loc,
                           pcl::KdTreeFLANN<PointType>& kdtree,
                           pcl::PointCloud<PointType>::Ptr global_map,
                           const StatusManifold& status) {
    
    // std::lock_guard<std::mutex> lock_match(global_map_match_mtx);
    
    pcl::PointCloud<PointType>::Ptr frame_glob (new pcl::PointCloud<PointType>);
    Eigen::Affine3f aff;
    aff.linear() = nomial_status.R.cast<float>();
    aff.translation() = nomial_status.p.cast<float>();
    pcl::transformPointCloud(*frame_loc, *frame_glob, aff, true);
    
    std::vector<float> dist_sq(5);
    std::vector<int> idx(5);
    for (size_t i = 0; i < frame_glob->size(); i ++) {
        if (kdtree.nearestKSearch(frame_glob->points[i], 5, idx, dist_sq)) {
            Eigen::Matrix<double, 3, 5> m;
            for (size_t j = 0; j < 5; j++) {
                m.block(0, j, 3, 1) = V3D(global_map->points[idx[j]].x,
                                          global_map->points[idx[j]].y,
                                          global_map->points[idx[j]].z);
            }
            
            PointType mean_point;
            Eigen::SelfAdjointEigenSolver<M3D> solver = pca_feature(m, mean_point);
            double ratio = solver.eigenvalues()[2] / (solver.eigenvalues()[1] + 1e-5);
            V3D cur_world(frame_glob->points[i].x, frame_glob->points[i].y, frame_glob->points[i].z);
            
            // line feature assign
            // std::cout << ratio << " " << std::endl;
            if (ratio > 4) {
                EdgeFeature fea;
                fea.direction = solver.eigenvectors().col(2);
                V3D map_point(mean_point.x, mean_point.y, mean_point.z);
                fea.p_loc = V3D(frame_loc->points[i].x, frame_loc->points[i].y, frame_loc->points[i].z);
                fea.residual = edge_residual(fea.direction, cur_world, map_point);
                if (fea.residual.squaredNorm() < 0.25) {
                    edge_buf.push_back(fea);
                }
            }
            // planar
            if (ratio < 4) {
                PlanarFeature fea;
                fea.normal = solver.eigenvectors().col(0);
                fea.p_loc = V3D(frame_loc->points[i].x, frame_loc->points[i].y, frame_loc->points[i].z);
                fea.residual = planar_residual(fea.normal, cur_world, V3D(mean_point.x, mean_point.y, mean_point.z));
                if (fea.residual < 0.5) {
                    planar_buf.push_back(fea);
                }
            }
        }
        
    }
}

Eigen::SelfAdjointEigenSolver<M3D> KF::pca_feature(Eigen::Matrix<double, 3, 5>& m, PointType& p_mean) {
    V3D mean = m.rowwise().mean();
        m = m.colwise() - mean;
        M3D res = m * m.transpose() / m.cols();
        Eigen::SelfAdjointEigenSolver<M3D> sol(res);
        
        p_mean.x = mean.x();
        p_mean.y = mean.y();
        p_mean.z = mean.z();
        
        return sol;
}

V3D KF::edge_residual(const V3D& edge, const V3D& p_loc, const V3D& center) {
    return asymmetric(edge) * ( p_loc - center);
}

double KF::planar_residual(const V3D& normal, const V3D& p_loc, const V3D& center) {
    return normal.transpose() * (p_loc - center);
}

Eigen::MatrixXd KF::diag_inv(Eigen::MatrixXd& m) {
    int col = m.cols();
    int row = m.rows();
    assert(col == row);

    Eigen::MatrixXd ret;
    ret.resizeLike(m);
    ret.setZero();

    for (size_t i = 0; i < col; i++) {
        ret(i, i) = 1 / (m(i, i) + 1e-8);
    }
    return ret;
}


