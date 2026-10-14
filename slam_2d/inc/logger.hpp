#include <iostream>
#include <fstream>
#include <format>
#include <chrono>
#include <filesystem>

using namespace std::chrono;
class Logger {
public:
    enum Level {
        INFO, WARNING, ERROR, SUCCESS  
    };
    std::string return_current_time_and_date() {
        auto now = std::chrono::system_clock::now();
        time_t in_time_t = system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
        return ss.str();
    }
    Logger(const std::string& folder = "../log/") {
        if (!std::filesystem::is_directory(folder)) {
            std::filesystem::create_directory(folder);
        }
        std::string file_name = folder + return_current_time_and_date() + ".log";
        file_out.open(file_name);
        if (file_out.is_open()) {
            std::cout << "create " + file_name + " successfully" << std::endl;
        }
    }
    void log(const std::string& words, const Level level) {
        std::string color_code;
        std::string prefix;
        switch (level) {
            case INFO:
                color_code = "";
                prefix = "INFO";
                break;
            case WARNING:
                color_code = "\033[33m";
                prefix = "WARNING";
                break;
            case ERROR:
                color_code = "\033[31m";
                prefix = "ERROR";
                break;
            case SUCCESS:
                color_code = "\033[32m";
                prefix = "SUCCESS";
                break;
            default:
                break;
        }
        auto now = std::chrono::system_clock::now();
        std::stringstream ss;
        time_t t = std::chrono::system_clock::to_time_t(now);
        ss << std::put_time(std::localtime(&t), "|%H%M%S.");
        auto m = time_point_cast<nanoseconds>(time_point_cast<seconds>(now));
        auto d_ms = std::chrono::duration_cast<milliseconds>(now - m);
        ss << d_ms.count();
        std::cout << ss.str() << '|' << color_code << prefix << '|' << "\033[0m"<< words << std::endl;
        file_out << ss.str() << '|' << prefix << '|' << words << std::endl;
        if (level == ERROR) {
            throw std::runtime_error(words);
        }
    }
private:
    std::ofstream file_out;   
};
