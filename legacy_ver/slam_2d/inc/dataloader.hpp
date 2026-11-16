#pragma once

#include <mutex>
#include <condition_variable>
#include <deque>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>
#include <nlohmann/json_fwd.hpp>

#include <serial_boost.hpp>

using PointType = pcl::PointXYZINormal;
using V3D = Eigen::Vector3d;
using M3D = Eigen::Matrix3d;

struct IMUMsg {
    uint64_t timestamp;  //us
    Eigen::Vector3d angular_velocity;  // rad/s
    Eigen::Vector3d linear_acceleration;  // m/s^2
};

struct LiDARMsg {
    uint64_t starttime;  // us, global timestamp
    uint64_t endtime;
    pcl::PointCloud<PointType>::Ptr cloud_ptr;  // curvature represent relative time.
};

class DataLoader {    
public:
    DataLoader(const nlohmann::json& loader_config);
    void start_receiving(const bool lidar_enb=false, const bool imu_enb=false);
    void stop_receiving();
private:
    #pragma pack(1)
    struct IMUFrame {
        uint8_t head_1 = 0xAA;
        uint8_t head_2 = 0x55;
        uint32_t timestamp;
        int16_t accel_xyz[3];
        int16_t gyro_xyz[3];
        uint8_t crc;
    };
    
    struct Point {
    	uint16_t distance;
    	uint8_t intensity;
    };
    
    struct LiDARFrame {
    	uint8_t head1 = 0xA5;
    	uint8_t head2 = 0x5A;
    	uint32_t timestamp;
    	uint16_t speed;
    	uint16_t start_angle;
    	uint16_t end_angle;
    	Point echo[32];
    	uint8_t crc;
    };
    #pragma pack()

    void imu_msg_cbk();
    void lidar_msg_cbk();
    uint8_t xor_crc(uint8_t* start, uint32_t size);    
    double deg2rad(double deg);
    
public:
    std::deque<IMUMsg> imu_q;
    std::deque<LiDARMsg> lidar_q;
    std::condition_variable cv;
    std::mutex mtx;
    std::atomic<bool> running = false;
    std::atomic<bool> data_ready = false;
private:
    // serial reading
    uint8_t byte;
    Serial port;
    LiDARMsg lidar_msg;
    IMUMsg imu_msg;
    // lidar cbk related
    double d_radian = 0.68;  // angular change between two pnts
    float d_time = 0.0;  // sec
    float accumulated_d_time = 0.0;
    double rot_freq = 0.0;
    double scan_freq = 0.0;
    size_t frame_size = 0;
    double valid_lb_sq = 0;  // valid lower bound square
    double valid_ub_sq = 0;
    size_t frame_rx_cnt = 0;
    // lower communication protocol
    double accel_lsb = 0.0;
    double gyro_lsb = 0.0;
    uint64_t timestamp_to_ns = 0;
};
