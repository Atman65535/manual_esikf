#include <deque>
#include <nlohmann/json.hpp>
#include <mutex>
#include <condition_variable>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "structure.hpp"
#include "so3_math.hpp"
#include "dataloader.hpp"

class LiDARProcessor {

public:
    LiDARProcessor(const nlohmann::json lidar_processor_cfg);
    bool check_before_undistort(const LiDARMsg& msg, bool& lidar_msg_ready);
    bool undistort(LiDARMsg& msg);
    pcl::PointCloud<PointType>::Ptr down_sampling(pcl::PointCloud<PointType>::Ptr original);
    bool init(pcl::PointCloud<PointType>::Ptr cloud);
    bool is_ready();
    void update_global_map(pcl::PointCloud<PointType>::Ptr new_cloud_glob);
    void forward_record(const IMUMsg& msg, const StatusManifold& status);
public:

    pcl::KdTreeFLANN<PointType> kdtree;
    std::mutex kdtree_mtx;
    
    pcl::PointCloud<PointType>::Ptr global_map;
    std::condition_variable cv_global_map_update;
    std::mutex global_map_mtx;
private:
    std::deque<pcl::PointCloud<PointType>::Ptr> slide_window;
    std::deque<IMUMsg> imu_history;
    std::deque<StatusManifold> status_history;
    double filter_angle;
    size_t window_size;
    double vox_size = 0.02;

    std::atomic<bool> ready = false;
    pcl::VoxelGrid<PointType> vg;
};

