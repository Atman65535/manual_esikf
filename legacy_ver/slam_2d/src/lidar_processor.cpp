#include "lidar_processor.hpp"

LiDARProcessor::LiDARProcessor(const nlohmann::json lidar_processor_cfg) {
    filter_angle = 2 * std::numbers::pi / lidar_processor_cfg["feature_points"].get<double>();
    window_size = lidar_processor_cfg["window_size"];
    vox_size = lidar_processor_cfg["vox_size"];    
    global_map = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    ready = false;
}

// call after kf forward
void LiDARProcessor::forward_record(const IMUMsg& msg, const StatusManifold& status) {
    imu_history.push_back(msg);
    status_history.push_back(status);
}

bool LiDARProcessor::check_before_undistort(const LiDARMsg& msg, bool& lidar_msg_ready) {
   if (imu_history.empty() || msg.starttime < imu_history.front().timestamp) {      
       if (!imu_history.empty() && msg.starttime < imu_history.front().timestamp) {
           lidar_msg_ready = false;
       }
       return false;
   }
   return true;
}

// change inplace
bool LiDARProcessor::undistort(LiDARMsg& lidar_msg) {
    pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
    bool ret = true;
    // compensate 
    size_t idx_start;
    // find target msg / status segment
    for (idx_start = 0; idx_start < imu_history.size()-1; idx_start++) {
        if (imu_history[idx_start].timestamp <= lidar_msg.starttime &&
            imu_history[idx_start + 1].timestamp > lidar_msg.starttime) {
            break;
        }
    }
    imu_history.erase(imu_history.begin(), imu_history.begin() + idx_start);
    status_history.erase(status_history.begin(), status_history.begin() + idx_start);
    if (imu_history.size() < 5) {
        std::cout << imu_history.size() << std::endl;
        throw std::logic_error("LiDAR Processor: wrong start and end time");
    }
    assert(imu_history.size() == status_history.size());
            
    auto&& it_p = lidar_msg.cloud_ptr->end();
    auto&& imu_prev = imu_history.end() - 1;
    auto&& status_prev = status_history.end() - 1;
    StatusManifold last_status = status_history.back();
    StatusManifold point_status;

    uint64_t start_time = imu_history.front().timestamp;
    double lidar_start_time = (double)(lidar_msg.starttime - start_time) / 1e9;
    
    while (true) {
        double dt = (lidar_start_time + it_p->curvature) - (double)(imu_prev->timestamp - start_time)/1e9;
        StatusTangent increment;
        increment.dtheta = dt * (imu_prev->angular_velocity - status_prev->bw);
        increment.dp = dt * status_prev->v;
        increment.dv = status_prev->R * (imu_prev->linear_acceleration - status_prev->ba) * dt + status_prev->g * dt;
        point_status = boxplus(*status_prev, increment);
        V3D p_i (it_p->x, it_p->y, it_p->z);
        p_i = last_status.R.transpose() * ((point_status.R * p_i + point_status.p) - last_status.p);
        it_p->x = p_i.x();
        it_p->y = p_i.y();
        it_p->z = p_i.z();
        ret = pcl::isFinite(*it_p);
        if (p_i.squaredNorm() > 0.04) {
            cloud->push_back(*it_p);
        } 

        if (it_p == lidar_msg.cloud_ptr->begin()) {
            lidar_msg.cloud_ptr = cloud;
            break;
        }
        it_p --;
        if (lidar_msg.starttime + it_p->curvature < imu_prev->timestamp) {
            imu_prev--;
            status_prev--;
        }
    }
    return ret;
}

// this function might return too less points.
pcl::PointCloud<PointType>::Ptr LiDARProcessor::down_sampling(pcl::PointCloud<PointType>::Ptr original) {
    pcl::PointCloud<PointType>::Ptr filtered (new pcl::PointCloud<PointType>);
    bool find = false;
    V3D start_point;
    
    for (size_t i = 1; i < original->size() - 1; i ++) {
        V3D cur(original->points[i].x, original->points[i].y, original->points[i].z);
        V3D last (original->points[i-1].x, original->points[i-1].y, original->points[i-1].z);
        V3D next (original->points[i + 1].x, original->points[i+1].y, original->points[i+1].z);
        if (find == false) {
            if ((last - cur).squaredNorm() < 0.04 && (next - cur).squaredNorm() < 0.04) {
                find = true;
                filtered->push_back(original->points[i]);
                start_point = cur;
            }
        }
        else {
            if (cur.normalized().transpose() * start_point.normalized() < std::cos(filter_angle)) {
                find = false;
            }
        }
    }
    return filtered;
}
bool LiDARProcessor::init(pcl::PointCloud<PointType>::Ptr cloud) {
    slide_window.push_back(cloud);

    if (slide_window.size() >= window_size) {
        global_map->clear();
        for (int i = 0; i < slide_window.size(); i ++) {
            *global_map += *slide_window[i];
        }
        vg.setInputCloud(global_map);
        vg.setLeafSize(vox_size, vox_size, vox_size);

        vg.filter(*global_map);
        kdtree.setInputCloud(global_map);
        ready = true;
    }
    return ready;
}

bool LiDARProcessor::is_ready() {
    return ready;
}

void LiDARProcessor::update_global_map(pcl::PointCloud<PointType>::Ptr new_cloud_glob) {
    // *global_map += *new_cloud_glob;
    // global_map->clear();
    // slide_window.push_back(new_cloud_glob);
    // if (slide_window.size() > window_size) {
    //     slide_window.pop_front();
    // }
    // for (size_t i = 0; i < slide_window.size(); i++) {
    //     *global_map += *slide_window[i];
    // }
    // vg.setInputCloud(global_map);
    // vg.setLeafSize(vox_size, vox_size, vox_size);

    // vg.filter(*global_map);
    kdtree.setInputCloud(global_map);
}

