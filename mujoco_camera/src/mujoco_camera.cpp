#include "mujoco_camera.h"

#include "bitbot_kernel/device/device_factory.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace bitbot {

MujocoDepthFrame::MujocoDepthFrame(int width, int height, std::vector<float> depth_m)
    : width_(width), height_(height), depth_m_(std::move(depth_m)) {}

float MujocoDepthFrame::get_distance(int x, int y) const {
  if (x < 0 || y < 0 || x >= width_ || y >= height_) {
    return 0.0f;
  }

  const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);
  if (index >= depth_m_.size()) {
    return 0.0f;
  }

  return depth_m_[index];
}

MujocoCamera::MujocoCamera(const pugi::xml_node& device_node)
    : MujocoDevice(device_node) {
  basic_type_ = static_cast<uint32_t>(BasicDeviceType::SENSOR);
  type_ = kMujocoCameraDeviceType;
    monitor_header_.headers = { "depth_center_mm", "min_depth_mm", "max_depth_mm" };
    monitor_data_.resize(monitor_header_.headers.size(), 0.0f);

  sensor_name_ = device_node.attribute("sensor").as_string();
  width_ = device_node.attribute("width").as_int(width_);
  height_ = device_node.attribute("height").as_int(height_);
    fps_ = device_node.attribute("fps").as_float(fps_);
  max_distance_m_ = device_node.attribute("max_distance").as_float(max_distance_m_);
  preview_enabled_ = device_node.attribute("preview").as_bool(false);
  preview_scale_ = device_node.attribute("preview_scale").as_int(preview_scale_);

  if (sensor_name_.empty()) {
    sensor_name_ = device_node.attribute("name").as_string();
  }

  if (width_ <= 0) {
    width_ = 160;
  }
  if (height_ <= 0) {
    height_ = 120;
  }
  if (preview_scale_ <= 0) {
    preview_scale_ = 1;
  }
    if (fps_ <= 0.0f) {
      fps_ = 30.0f;
    }

  preview_window_name_ = "MuJoCo depth: " + sensor_name_;
}

void MujocoCamera::UpdateModel(const mjModel* m, mjData* d) {
  model_ = m;
  data_ = d;
  ready_ = false;

  if (model_ == nullptr || data_ == nullptr || sensor_name_.empty()) {
    return;
  }

  sensor_id_ = mj_name2id(model_, mjOBJ_SENSOR, sensor_name_.c_str());
  if (sensor_id_ < 0) {
    std::cerr << "[MujocoCamera] sensor not found: " << sensor_name_ << std::endl;
    return;
  }

  sensor_adr_ = model_->sensor_adr[sensor_id_];
  sensor_dim_ = model_->sensor_dim[sensor_id_];

  const int expected_dim = width_ * height_;
  if (sensor_dim_ < expected_dim) {
    std::cerr << "[MujocoCamera] sensor dim mismatch, sensor=" << sensor_name_
              << ", dim=" << sensor_dim_
              << ", expected at least=" << expected_dim << std::endl;
    return;
  }

  ready_ = true;
  std::cout << "[MujocoCamera] initialized: sensor=" << sensor_name_
            << ", size=" << width_ << "x" << height_
              << ", fps=" << fps_
            << ", dim=" << sensor_dim_ << std::endl;
}

void MujocoCamera::Input(const mjModel* m, mjData* d) {
  model_ = m;
  data_ = d;
}

void MujocoCamera::Output(const mjModel*, mjData*) {}

void MujocoCamera::UpdateRuntimeData() {
  std::lock_guard<std::mutex> lock(frame_mutex_);
  monitor_data_[0] = latest_center_mm_;
  monitor_data_[1] = latest_min_mm_;
  monitor_data_[2] = latest_max_mm_;
}

void MujocoCamera::UpdateFrame() {
  if (!ready_ || model_ == nullptr || data_ == nullptr || sensor_adr_ < 0) {
    return;
  }

  const int pixel_count = width_ * height_;
  std::vector<float> depth_m(static_cast<size_t>(pixel_count), 0.0f);

  const mjtNum* src = data_->sensordata + sensor_adr_;

  float min_depth_m = max_distance_m_;
  float max_depth_m = 0.0f;

  for (int i = 0; i < pixel_count; ++i) {
    const float value = static_cast<float>(src[i]);

    if (std::isfinite(value) && value > 0.0f && value <= max_distance_m_) {
      depth_m[static_cast<size_t>(i)] = value;
      min_depth_m = std::min(min_depth_m, value);
      max_depth_m = std::max(max_depth_m, value);
    } else {
      depth_m[static_cast<size_t>(i)] = 0.0f;
    }
  }

  const float center_depth_m = depth_m[static_cast<size_t>(height_ / 2) * static_cast<size_t>(width_) +
                                       static_cast<size_t>(width_ / 2)];

  std::lock_guard<std::mutex> lock(frame_mutex_);
  latest_depth_frame_ = MujocoDepthFrame(width_, height_, std::move(depth_m));
  latest_center_mm_ = center_depth_m * 1000.0f;
  latest_min_mm_ = (max_depth_m > 0.0f) ? min_depth_m * 1000.0f : 0.0f;
  latest_max_mm_ = max_depth_m * 1000.0f;

  if (preview_enabled_) {
    if (!preview_started_) {
      std::cout << "[MujocoCamera] depth preview window started: " << preview_window_name_ << std::endl;
      preview_started_ = true;
    }

    const auto& frame = latest_depth_frame_.value();
    cv::Mat gray(height_, width_, CV_8UC1);
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        const float depth = frame.get_distance(x, y);
        const float normalized = depth > 0.0f ? std::clamp(depth / max_distance_m_, 0.0f, 1.0f) : 1.0f;
        gray.at<unsigned char>(y, x) = static_cast<unsigned char>((1.0f - normalized) * 255.0f);
      }
    }

    if (preview_scale_ > 1) {
      cv::Mat resized;
      cv::resize(gray, resized, cv::Size(width_ * preview_scale_, height_ * preview_scale_), 0, 0, cv::INTER_NEAREST);
      cv::imshow(preview_window_name_, resized);
    } else {
      cv::imshow(preview_window_name_, gray);
    }
    cv::waitKey(1);
  }
}

std::optional<MujocoDepthFrame> MujocoCamera::GetDepthFrame() const {
  std::lock_guard<std::mutex> lock(frame_mutex_);
  return latest_depth_frame_;
}

}  // namespace bitbot

namespace {
bitbot::DeviceRegistrar<bitbot::MujocoDevice, bitbot::MujocoCamera> kMujocoCameraRegistrar(
    bitbot::kMujocoCameraDeviceType,
    "MujocoCamera");
}
