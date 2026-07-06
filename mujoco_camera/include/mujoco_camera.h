#pragma once

#include "bitbot_mujoco/device/mujoco_device.hpp"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace bitbot {

inline constexpr uint32_t kMujocoCameraDeviceType = 11006;

class MujocoDepthFrame {
public:
  MujocoDepthFrame() = default;
  MujocoDepthFrame(int width, int height, std::vector<float> depth_m);

  float get_distance(int x, int y) const;
  int get_width() const { return width_; }
  int get_height() const { return height_; }
  bool empty() const { return depth_m_.empty(); }

private:
  int width_ = 0;
  int height_ = 0;
  std::vector<float> depth_m_;
};

class MujocoCamera final : public MujocoDevice {
public:
  explicit MujocoCamera(const pugi::xml_node& device_node);
  ~MujocoCamera() = default;

  bool IsReady() const { return ready_; }
  int GetWidth() const { return width_; }
  int GetHeight() const { return height_; }
  float GetDepthScale() const { return 1.0f; }
  float GetFps() const { return fps_; }

  void UpdateFrame();

  std::optional<MujocoDepthFrame> GetDepthFrame() const;

private:
  void UpdateModel(const mjModel* m, mjData* d) override;
  void Input(const mjModel* m, mjData* d) override;
  void Output(const mjModel* m, mjData* d) override;
  void UpdateRuntimeData() override;

private:
  std::string sensor_name_;
  int width_ = 160;
  int height_ = 120;
  float fps_ = 30.0f;
  float max_distance_m_ = 3.0f;
  bool preview_enabled_ = false;
  bool preview_started_ = false;
  int preview_scale_ = 2;
  std::string preview_window_name_;
    float latest_center_mm_ = 0.0f;
    float latest_min_mm_ = 0.0f;
    float latest_max_mm_ = 0.0f;

  const mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  int sensor_id_ = -1;
  int sensor_adr_ = -1;
  int sensor_dim_ = 0;

  bool ready_ = false;

  mutable std::mutex frame_mutex_;
  std::optional<MujocoDepthFrame> latest_depth_frame_;
};

}  // namespace bitbot
