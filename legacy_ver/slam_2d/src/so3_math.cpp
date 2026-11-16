#include "so3_math.hpp"

inline M3D asymmetric(const V3D vector) {
    M3D m ({
           {0, -vector.z(), vector.y()},
           {vector.z(), 0, -vector.x()},
           {-vector.y(), vector.x(), 0}});
    return m;
}

inline V3D vee_vector(const M3D m) {
    if (m != -m.transpose()) {
        std::cout << "vee_vector: wrong m type!" << std::endl << m << std::endl; 
        throw std::format_error("wrong");
    }
    return V3D({m(2, 1), m(0, 2), m(1, 0)});
}

inline M3D Exp(const V3D lie_algebra) {
    double theta = lie_algebra.norm();
    if (theta > 0.0001) { 
        M3D hat_u = asymmetric(lie_algebra / theta);
        return (M3D::Identity() + hat_u * sin(theta) + hat_u * hat_u * (1 - cos(theta)));
    }
    else {
        return M3D::Identity();
    }
}

// aborted
M3D rodrigus_formula(const V3D n, const double theta) {
    M3D R = cos(theta) * M3D::Identity() + (1. - cos(theta)) * n * n.transpose() + sin(theta) * asymmetric(n);
    return R;
}

V3D Log(const M3D &R)
{
    double theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
    V3D K(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));
    return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}


StatusManifold boxplus(const StatusManifold& s, const StatusTangent& t) {
    StatusManifold m;
    m.R = s.R * Exp(t.dtheta);
    m.p = s.p + t.dp;
    m.v = s.v + t.dv;
    m.ba = s.ba + t.dba;
    m.bw = s.bw + t.dbw;
    m.g = s.g + t.dg;
    return m;
}

StatusManifold boxplus(const StatusManifold& s, const Eigen::Matrix<double, 18, 1>& t) {
    StatusTangent e;
    e.dtheta = t.block(0, 0, 3, 1);
    e.dp = t.block(3, 0, 3, 1);
    e.dv = t.block(6, 0, 3, 1);
    e.dbw = t.block(9, 0, 3, 1);
    e.dba = t.block(12, 0, 3, 1);
    e.dg = t.block(15, 0, 3, 1);
    return boxplus(s, e);
}


// s - m
Eigen::Matrix<double, 18, 1> boxminus(const StatusManifold& s, const StatusManifold& m) {
    StatusTangent t;
    t.dtheta = Log(m.R.transpose() * s.R);
    t.dp = s.p - m.p;
    t.dv = s.v - m.v;
    t.dbw = s.bw - m.bw;
    t.dba = s.ba - m.ba;
    t.dg = s.g - m.g;
    Eigen::Matrix<double, 18, 1> ret;
    ret.block(0, 0, 3, 1) = t.dtheta;
    ret.block(3, 0, 3, 1) = t.dp;
    ret.block(6, 0, 3, 1) = t.dv;
    ret.block(9, 0, 3, 1) = t.dbw;
    ret.block(12, 0, 3, 1) =t.dba;
    ret.block(15, 0, 3, 1) = t.dg;
    return ret;
}


// refer to Vision SLAM 14 Lessons.
// usage: exp(\fai)exp(Jr \Delta \fai) = exp(\fai + \Delta fai)
M3D SO3_left_jacobian(const V3D main_part) {
    double theta = main_part.norm();
    if (theta > 0.001) {
    V3D unified = main_part / theta;
    return (std::sin(theta) / theta) * M3D::Identity()
           + (1 - std::sin(theta) / theta) * unified * unified.transpose()
           + ((1 - std::cos(theta)) / theta) * asymmetric(unified);
    }
    else {
        return M3D::Identity();
    }
}

M3D SO3_left_jacobian_inv(const V3D main_part) {
    double theta = main_part.norm();
    if (theta > 0.001) {
    V3D unified = main_part / theta;
    return (theta / 2) / std::tan(theta / 2) * M3D::Identity()
            + (1 - theta / 2 / tan(theta / 2)) * unified * unified.transpose()
            - theta / 2 * asymmetric(unified);
    }
    else {
        return M3D::Identity();
    }
}

M3D SO3_right_jacobian(const V3D main_part) {
    return SO3_left_jacobian(main_part).transpose();
}

M3D SO3_right_jacobian_inv(const V3D main_part) {
    return SO3_left_jacobian_inv(main_part).transpose();
}


void from_ego_to_world(pcl::PointCloud<PointType>::Ptr cloud, const M3D& R, const V3D& p) {
    Eigen::Affine3f aff;
    aff.linear() = R.cast<float>();
    aff.translation() = p.cast<float>();
    pcl::transformPointCloud(*cloud, *cloud, aff);
}

void from_ego_to_world(pcl::PointCloud<PointType>::Ptr loc_cloud,
                       pcl::PointCloud<PointType>::Ptr glob_cloud,
                       const M3D& R_ego_to_world, const V3D& p) {
    Eigen::Affine3f aff;
    aff.linear() = R_ego_to_world.cast<float>();
    aff.translation() = p.cast<float>();
    pcl::transformPointCloud(*loc_cloud, *glob_cloud, aff);
}

V3D from_world_to_ego_v3d(const PointType& pnt, const M3D& R_ego_to_world, const V3D& p) {
    PointType p_;
    V3D org(pnt.x, pnt.y, pnt.z);
    org = org - p;
    org = R_ego_to_world.transpose() * org;
    return org;
}
