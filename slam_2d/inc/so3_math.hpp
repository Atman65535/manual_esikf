# pragma once
#include "structure.hpp"
#include <Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>
#include <pcl/common/transforms.h>
#include <format>

M3D asymmetric(const V3D vector);
V3D vee_vector(const M3D m);
M3D rodrigus_formula(const V3D n, const double theta);

M3D Exp(const V3D lie_algebra);
V3D Log(const M3D& R);

StatusManifold boxplus(const StatusManifold& s, const StatusTangent& t);
StatusManifold boxplus(const StatusManifold& s, const Eigen::Matrix<double, 18, 1>& t);
Eigen::Matrix<double, 18, 1> boxminus(const StatusManifold& s, const StatusManifold& m);

// BCH
M3D SO3_left_jacobian(const V3D main_part);
M3D SO3_left_jacobian_inv(const V3D main_part);
M3D SO3_right_jacobian(const V3D main_part);
M3D SO3_right_jacobian_inv(const V3D main_part);
void from_ego_to_world(pcl::PointCloud<PointType>::Ptr cloud, const M3D& R, const V3D& p);

void from_ego_to_world(pcl::PointCloud<PointType>::Ptr loc_cloud,
                       pcl::PointCloud<PointType>::Ptr glob_cloud,
                       const M3D& R_ego_to_world, const V3D& p);

V3D from_world_to_ego_v3d(const PointType& pnt, const M3D& R_ego_to_world, const V3D& p);
