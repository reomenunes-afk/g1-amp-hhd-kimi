#pragma once

#include "device/unitree_device.hpp"

#include <librealsense2/rs.hpp>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace bitbot {

  struct DepthFramePacket {
    bool valid = false;
    uint64_t frame_seq = 0;
    double timestamp_ms = 0.0;
    int width = 0;
    int height = 0;
    std::optional<rs2::depth_frame> depth_frame;
    std::optional<rs2::video_frame> color_frame;
  };

  class UnitreeCamera final : public UnitreeDevice {
  public:
    UnitreeCamera(pugi::xml_node const& device_node);
    ~UnitreeCamera();

    bool IsReady() const { return initialized_; }
    int GetWidth() const { return static_cast<int>(width_); }
    int GetHeight() const { return static_cast<int>(height_); }
    float GetDepthScale() const { return depth_scale_; }

    void UpdateFrame();

    std::optional<rs2::depth_frame> GetDepthFrame() const;
    std::optional<rs2::video_frame> GetColorFrame() const;

    DepthFramePacket GetLatestDepthPacket() const;
    std::vector<DepthFramePacket> GetDepthHistory(size_t frame_count = 8) const; // oldest -> newest
    bool HasDepthHistory(size_t required_frames = 8) const;
    bool SaveLatestDepthPGM(const std::string& path, float max_dist_m = 3.0f) const;

  private:
    virtual void Input(const IOType& IO) final;
    virtual IOType Output() final;
    virtual void UpdateRuntimeData() final;

    void InitCamera();

  private:
    static constexpr size_t kDepthHistorySize = 8;

    rs2::pipeline pipeline_;
    rs2::config config_;

    mutable std::mutex frame_mutex_;
    std::optional<rs2::depth_frame> depth_frame_;
    std::optional<rs2::video_frame> color_frame_;
    std::deque<DepthFramePacket> depth_history_;

    uint64_t latest_frame_seq_ = 0;
    double latest_timestamp_ms_ = 0.0;

    float depth_scale_ = 0.001f;
    double width_ = 640.0;
    double height_ = 480.0;
    double fps_ = 30.0;
    bool initialized_ = false;

    float latest_center_mm_ = 0.0f;
    float latest_min_mm_ = 0.0f;
    float latest_max_mm_ = 0.0f;
  };

}  // namespace bitbot