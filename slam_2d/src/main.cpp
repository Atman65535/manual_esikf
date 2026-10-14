#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <nlohmann/json.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <chrono>

#include "dataloader.hpp"
#include "so3_math.hpp"
#include "lidar_processor.hpp"
#include "esikf.hpp"
#include "logger.hpp"

using namespace std::chrono_literals;

std::atomic<bool> system_running = false;

bool lidar_msg_ready = false;
bool imu_msg_ready = false;
IMUMsg imu_msg;
LiDARMsg lidar_msg;

std::mutex kf_posterior_mtx;
std::condition_variable kf_post_cv;
std::atomic<bool> posterior_ready = false;

std::shared_mutex lidar_map_rw_mutex;

std::mutex lidar_map_cv_mutex;
std::condition_variable lidar_map_cv;
std::atomic<bool> lidar_map_ready = false;
std::atomic<bool> lidar_map_vis_ready = false;

pcl::PointCloud<PointType>::Ptr cur_glob_increment (new pcl::PointCloud<PointType>);
StatusManifold posterior_status;  // output

Logger logger;
std::atomic<int> kf_iter_time, map_update_time;

uint64_t imu_last_stamp;
bool data_sync(DataLoader& loader) {
    bool ret = false;
    std::unique_lock<std::mutex> lock(loader.mtx);
    loader.cv.wait(lock, [&loader]{return !loader.imu_q.empty();});
    if (!loader.imu_q.empty() && !imu_msg_ready) {
        imu_msg_ready = true;
        imu_msg = loader.imu_q.front();
        imu_last_stamp = imu_msg.timestamp;
        loader.imu_q.pop_front();
        ret = true;
        if (!loader.lidar_q.empty() && !lidar_msg_ready) {
            lidar_msg = loader.lidar_q.front();
            if (imu_last_stamp > lidar_msg.endtime) {
                loader.lidar_q.pop_front();
                ret = true;
                lidar_msg_ready = true;
            }
        }
    }
    
    lock.unlock();
    return ret;
}

void init_components(KF& kf, LiDARProcessor& handler, DataLoader& loader) {
    bool kf_ready = false;
    bool map_ready = false;
    
    while (system_running) {
        if (data_sync(loader)) {
            if (imu_msg_ready && !kf_ready) {
                kf_ready = kf.init(imu_msg);
                imu_msg_ready = false;
            }
            else if (lidar_msg_ready) {
                lidar_msg_ready = false;
            }
        }
        if (kf_ready) {
            break;
        }
    }
    logger.log("IMU Init", Logger::INFO);
    std::stringstream ss;
    ss << kf.get_status().R;
    logger.log(ss.str(), Logger::INFO);
    StatusManifold status = kf.get_status();
    while (system_running) {
        if (data_sync(loader)) {
            if (imu_msg_ready) {
                imu_msg_ready = false;
                // Note that here is no need to wrap lidar
                // service in the one imu use. But for consistency,
                // we(I) implement this style.
                if (lidar_msg_ready && !map_ready) {
                    auto cloud = lidar_msg.cloud_ptr;
                    from_ego_to_world(cloud, status.R, status.p);
                    map_ready = handler.init(cloud);
                    lidar_msg_ready = false;
                }
            }
        }
        if (map_ready) {
            break;
        }
    }
    logger.log("LiDAR map init!", Logger::INFO);
}

void get_data_task(DataLoader& loader) {
    loader.start_receiving(true, true);
}

void kf_update_task(KF& kf, LiDARProcessor& lidar_handler, DataLoader& loader) {
    
    if (!kf.is_ready() || !lidar_handler.is_ready()) {
        if (!kf.is_ready()) {
            throw std::runtime_error("kf not initialized");
        }
        else if (!lidar_handler.is_ready()) {
            throw std::runtime_error("lidar not initialized");
        }
    }
    while (system_running) {
        if (data_sync(loader)) {
            if (imu_msg_ready) {
                imu_msg_ready = false;
                kf.forward_propagation(imu_msg); 
                auto st = kf.get_status();
                lidar_handler.forward_record(imu_msg, st);
                // logger.log ("forward", Logger::INFO);
                // discard if the check set false
                if (lidar_msg_ready && lidar_handler.check_before_undistort(lidar_msg, lidar_msg_ready)) {
                    lidar_msg_ready = false;                
                    if (!lidar_handler.undistort(lidar_msg)) {
                        logger.log("nan detected during undistort", Logger::ERROR);
                    }
                    pcl::copyPointCloud(*lidar_msg.cloud_ptr, *cur_glob_increment);
                    lidar_msg.cloud_ptr = lidar_handler.down_sampling(lidar_msg.cloud_ptr);
                    std::unique_lock<std::mutex> lock(lidar_map_cv_mutex);
                    lidar_map_cv.wait(lock, []{return lidar_map_ready == true;});
                    lock.unlock();
                
                    std::shared_lock<std::shared_mutex> read_lock(lidar_map_rw_mutex);
                    std::unique_lock<std::mutex> kf_lock(kf_posterior_mtx);

                    std::chrono::time_point<std::chrono::system_clock, nanoseconds> now = std::chrono::system_clock::now();
                    kf.measurament_update(lidar_handler.global_map, lidar_handler.kdtree, lidar_msg.cloud_ptr);
                    posterior_status = kf.get_status();
                    auto finish = std::chrono::system_clock::now();
                    kf_iter_time = std::chrono::duration_cast<milliseconds>(finish - now).count();
                    read_lock.unlock();
                    kf_lock.unlock();
                    posterior_ready = true;
                    kf_post_cv.notify_one();
                }
            }
        }
    }
}

void map_update_task(LiDARProcessor& handler) {
    if (handler.is_ready()) {
        lidar_map_ready = true;
    }
    while (system_running) {
        std::unique_lock<std::mutex> kf_post_lock(kf_posterior_mtx);
        kf_post_cv.wait(kf_post_lock, []{return posterior_ready == true;});
        posterior_ready = false;
        kf_post_lock.unlock();
        
        std::unique_lock<std::mutex> map_update_lock(lidar_map_cv_mutex);
        lidar_map_ready = false;
        map_update_lock.unlock();
        
        std::unique_lock<std::shared_mutex> map_write_lock(lidar_map_rw_mutex);
        auto now = std::chrono::system_clock::now();
        from_ego_to_world(cur_glob_increment, posterior_status.R, posterior_status.p);
        // std::cout << posterior_status.R.determinant() << std::endl;
        handler.update_global_map(cur_glob_increment);
        auto finish = std::chrono::system_clock::now();
        map_update_time = std::chrono::duration_cast<milliseconds>(finish - now).count();
        map_write_lock.unlock();
        lidar_map_ready = true;
        lidar_map_vis_ready = true;
        lidar_map_cv.notify_all();
    }
}

int main() {
    std::ifstream file_in("/home/atman/a_workspace/slam_lab/slam_software/config/config.json");
    nlohmann::json cfg = nlohmann::json::parse(file_in);

    DataLoader loader(cfg["loader_cfg"]);
    KF kf(cfg["kf_cfg"]);
    LiDARProcessor lidar_handler(cfg["lidar_processor_cfg"]);
    pcl::visualization::PCLVisualizer viewer;

    system_running = true;
    std::thread data_receiving(get_data_task, std::ref(loader));
    init_components(kf, lidar_handler, loader);
    
    // int imu_cnt = 0;
    // int lidar_cnt = 0;
    // while (true) {
    //     if (data_sync(loader)) {
    //         if (imu_msg_ready) {
    //             std::cout << imu_cnt ++ << "\r";
    //             imu_msg_ready = false;
    //         }
    //         if (lidar_msg_ready) {
    //             lidar_msg_ready = false;
    //             std::cout << "          " << lidar_cnt++ << "\r";
    //         }
    //         std::flush(std::cout);
    //     }
    // } 
    
    std::thread kf_updating(kf_update_task, std::ref(kf), std::ref(lidar_handler), std::ref(loader));
    std::thread map_updating(map_update_task, std::ref(lidar_handler));

    pcl::PointXYZ last_pos, cur_pos;
    last_pos.x = 0;
    last_pos.y = 0;
    last_pos.z = 0;
    int line_idx = 0;
    viewer.addCoordinateSystem(1.0, "ego");
    viewer.addPointCloud<PointType>(lidar_handler.global_map, "cloud");
    while (!viewer.wasStopped()) {
        viewer.spinOnce(10);
        if (lidar_map_vis_ready) {
            lidar_map_vis_ready = false;
        
            std::shared_lock<std::shared_mutex> update_lock(lidar_map_rw_mutex);
            pcl::visualization::PointCloudColorHandlerGenericField<PointType> color(lidar_handler.global_map, "z");
            viewer.updatePointCloud<PointType>(lidar_handler.global_map, color, "cloud");
            update_lock.unlock();
            
            std::unique_lock<std::mutex> kf_lock(kf_posterior_mtx);
            cur_pos.x = posterior_status.p.x();
            cur_pos.y = posterior_status.p.y();
            cur_pos.z = posterior_status.p.z();
            V3D velocity = posterior_status.v;
            last_pos = cur_pos;
            Eigen::Affine3f aff;
            aff.linear() = posterior_status.R.cast<float>();
            aff.translation() = posterior_status.p.cast<float>();
            
            std::stringstream msg;
            char buf[100];
            msg << "kf update time: " << kf_iter_time << '/' << kf.iter << "ms" << "  |  map update time: " << map_update_time << "ms\n";
            sprintf(buf, "p: %.2f %.2f %.2f | v: %.2f %.2f %.2f | ba: %.2f %.2f %.2f | bw: %.2f %.2f %.2f | g: %.2f %.2f %.2f",
                    cur_pos.x, cur_pos.y, cur_pos.z,
                    velocity.x(), velocity.y(), velocity.z(),
                posterior_status.ba.x(), posterior_status.ba.y(), posterior_status.ba.z(),
                posterior_status.bw.x(), posterior_status.bw.y(), posterior_status.bw.z(),
                posterior_status.g.x(), posterior_status.g.y(), posterior_status.g.z());
            kf_lock.unlock();
            msg << "Status: " << buf;   
            logger.log(msg.str(), Logger::INFO);
            viewer.updateCoordinateSystemPose("ego", aff);
            viewer.addLine(last_pos, cur_pos, 0, 125, 125, "line" + std::to_string(line_idx++));
        }
        std::this_thread::sleep_for(5ms);
    }
    system_running = false;
    loader.stop_receiving();
    data_receiving.join();
    kf_updating.join();
    map_updating.join();
    return 0;
}
