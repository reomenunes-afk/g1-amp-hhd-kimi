#include "device/unitree_camera.h"

#include <algorithm>
#include <fstream>
#include <vector>

namespace bitbot {

  UnitreeCamera::UnitreeCamera(pugi::xml_node const& device_node)
      : UnitreeDevice(device_node) {
    basic_type_ = (uint32_t)BasicDeviceType::SENSOR;
    type_ = (uint32_t)UnitreeDeviceType::UNITREE_CAMERA;

    ConfigParser::ParseAttribute2d(width_, device_node.attribute("width"));
    ConfigParser::ParseAttribute2d(height_, device_node.attribute("height"));
    ConfigParser::ParseAttribute2d(fps_, device_node.attribute("fps"));

    monitor_header_.headers = { "depth_center_mm", "min_depth_mm", "max_depth_mm" };
    monitor_data_.resize(monitor_header_.headers.size(), 0.0f);

    InitCamera();
  }

  UnitreeCamera::~UnitreeCamera() {
    if (initialized_) {
      pipeline_.stop();
    }
  }

  void UnitreeCamera::InitCamera() {
    try {
      config_.enable_stream(
          RS2_STREAM_DEPTH,
          static_cast<int>(width_),
          static_cast<int>(height_),
          RS2_FORMAT_Z16,
          static_cast<int>(fps_));
      config_.enable_stream(
          RS2_STREAM_COLOR,
          static_cast<int>(width_),
          static_cast<int>(height_),
          RS2_FORMAT_BGR8,
          static_cast<int>(fps_));

      rs2::pipeline_profile profile = pipeline_.start(config_);
      rs2::device dev = profile.get_device();
      rs2::depth_sensor ds = dev.first<rs2::depth_sensor>();
      if (ds) {
        depth_scale_ = ds.get_depth_scale();
      }

      initialized_ = true;
      logger_->info("RealSense camera initialized: {}x{} @ {}fps",
                    static_cast<int>(width_),
                    static_cast<int>(height_),
                    static_cast<int>(fps_));
    } catch (const rs2::error& e) {
      logger_->error("RealSense camera init failed: {}", e.what());
      initialized_ = false;
    }
  }

  void UnitreeCamera::UpdateFrame() {
  if (!initialized_) return;

  try {
    rs2::frameset new_frames;
    if (!pipeline_.poll_for_frames(&new_frames)) {
      return;
    }

    rs2::depth_frame new_depth = new_frames.get_depth_frame();
    rs2::video_frame new_color = new_frames.get_color_frame();
    if (!new_depth) {
      return;
    }

    DepthFramePacket pkt;
    pkt.valid = true;
    pkt.frame_seq = new_depth.get_frame_number();
    pkt.timestamp_ms = new_depth.get_timestamp();
    pkt.width = static_cast<int>(width_);
    pkt.height = static_cast<int>(height_);
    pkt.depth_frame = new_depth;
    if (new_color) {
      pkt.color_frame = new_color;
    }

    float center_dist = new_depth.get_distance(pkt.width / 2, pkt.height / 2);
    float min_dist = 10.0f;
    float max_dist = 0.0f;

    for (int y = 0; y < pkt.height; y += 10) {
      for (int x = 0; x < pkt.width; x += 10) {
        float dist = new_depth.get_distance(x, y);
        if (dist > 0.0f && dist < 10.0f) {
          min_dist = std::min(min_dist, dist);
          max_dist = std::max(max_dist, dist);
        }
      }
    }

    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      depth_frame_ = new_depth;
      color_frame_ = new_color ? std::optional<rs2::video_frame>(new_color) : std::nullopt;
      latest_frame_seq_ = pkt.frame_seq;
      latest_timestamp_ms_ = pkt.timestamp_ms;
      latest_center_mm_ = center_dist * 1000.0f;   // 改这里
      latest_min_mm_ = min_dist * 1000.0f;          // 改这里
      latest_max_mm_ = max_dist * 1000.0f;          // 改这里

      // std::cout << "[camera] frame ok center=" << latest_center_mm_
      //     << " min=" << latest_min_mm_
      //     << " max=" << latest_max_mm_ << std::endl;

      depth_history_.push_back(pkt);
      while (depth_history_.size() > kDepthHistorySize) {
        depth_history_.pop_front();
      }
    }
  } catch (const rs2::error& e) {
    logger_->warn("RealSense frame update failed: {}", e.what());
  }
}

  std::optional<rs2::depth_frame> UnitreeCamera::GetDepthFrame() const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return depth_frame_;
  }

  std::optional<rs2::video_frame> UnitreeCamera::GetColorFrame() const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return color_frame_;
  }

  DepthFramePacket UnitreeCamera::GetLatestDepthPacket() const {
    std::lock_guard<std::mutex> lock(frame_mutex_);

    DepthFramePacket pkt;
    pkt.valid = depth_frame_.has_value();
    pkt.frame_seq = latest_frame_seq_;
    pkt.timestamp_ms = latest_timestamp_ms_;
    pkt.width = static_cast<int>(width_);
    pkt.height = static_cast<int>(height_);
    pkt.depth_frame = depth_frame_;
    pkt.color_frame = color_frame_;
    return pkt;
  }

  std::vector<DepthFramePacket> UnitreeCamera::GetDepthHistory(size_t frame_count) const {
    std::lock_guard<std::mutex> lock(frame_mutex_);

    if (frame_count == 0 || depth_history_.empty()) {
      return {};
    }

    const size_t count = std::min(frame_count, depth_history_.size());
    const size_t start = depth_history_.size() - count;

    std::vector<DepthFramePacket> out;
    out.reserve(count);
    for (size_t i = start; i < depth_history_.size(); ++i) {
      out.push_back(depth_history_[i]);
    }
    return out;
  }

  bool UnitreeCamera::HasDepthHistory(size_t required_frames) const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return depth_history_.size() >= required_frames;
  }

  bool UnitreeCamera::SaveLatestDepthPGM(const std::string& path, float max_dist_m) const {
    DepthFramePacket pkt = GetLatestDepthPacket();
    if (!pkt.valid || !pkt.depth_frame.has_value()) {
      return false;
    }

    rs2::depth_frame depth = pkt.depth_frame.value();
    std::vector<unsigned char> img(pkt.width * pkt.height, 0);

    for (int y = 0; y < pkt.height; ++y) {
      for (int x = 0; x < pkt.width; ++x) {
        float d = depth.get_distance(x, y);
        unsigned char v = 0;
        if (d > 0.0f && d < max_dist_m) {
          v = static_cast<unsigned char>(255.0f * (1.0f - d / max_dist_m));
        }
        img[y * pkt.width + x] = v;
      }
    }

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
      return false;
    }

    ofs << "P5\n" << pkt.width << " " << pkt.height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(img.size()));
    return true;
  }

  void UnitreeCamera::Input(const IOType& IO) {}
  IOType UnitreeCamera::Output() { return IOType(); }
  void UnitreeCamera::UpdateRuntimeData() {
  std::lock_guard<std::mutex> lock(frame_mutex_);
  monitor_data_[0] = latest_center_mm_;
  monitor_data_[1] = latest_min_mm_;
  monitor_data_[2] = latest_max_mm_;
}

}  // namespace bitbot