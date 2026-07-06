#include "raycaster_src/RayCasterCamera.h"
#include "raycaster_src/RayNoise.hpp"

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Pose {
  double timestamp = 0.0;
  std::array<double, 3> pos{};
  std::array<double, 4> quat_xyzw{};
};

struct Args {
  fs::path trajectory;
  fs::path output;
  fs::path xml = fs::path("mujoco_camera_replay.xml");
  fs::path plugin = fs::path("../../lib/libsensor_raycaster.so");
  std::string body = "tracked_camera_mocap";
  std::string camera = "tracked_camera";
  std::string primary_sensor = "tracked_depth_raycaster";
  std::string secondary_sensor = "tracked_aux_depth_raycaster";
  int every = 1;
  int max_frames = 0;
  bool render_rgb = true;
  std::string render_size = "auto";
  double scale = 1.0;
  double z_offset = 0.0;
  std::string basis = "opencv_to_mujoco";
  std::string origin = "first";
  std::string origin_offset = "0 0 0";
  std::optional<double> min_depth;
  std::optional<double> max_depth;
  std::optional<double> depth_clip_min;
  std::optional<double> depth_clip_max;
  bool show_progress = true;
};

struct PluginConfig {
  std::map<std::string, std::string> kv;
};

struct DepthPack {
  std::vector<float> values;
  int width = 0;
  int height = 0;
};

[[noreturn]] void die(const std::string &msg) {
  throw std::runtime_error(msg);
}

std::vector<std::string> split_ws(const std::string &text) {
  std::istringstream iss(text);
  std::vector<std::string> out;
  std::string token;
  while (iss >> token) {
    out.push_back(token);
  }
  return out;
}

std::vector<double> parse_number_list(const std::string &text) {
  std::string cleaned = text;
  std::replace(cleaned.begin(), cleaned.end(), ',', ' ');
  std::istringstream iss(cleaned);
  std::vector<double> out;
  double v = 0.0;
  while (iss >> v) {
    out.push_back(v);
  }
  return out;
}

std::array<double, 3> parse_vec3(const std::string &text, const std::string &name) {
  auto values = parse_number_list(text);
  if (values.size() != 3) {
    die(name + " must contain exactly three numbers");
  }
  return {values[0], values[1], values[2]};
}

std::array<double, 9> parse_basis(const std::string &text) {
  if (text == "identity") {
    return {1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0};
  }
  if (text == "opencv_to_mujoco") {
    return {1.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, -1.0, 0.0};
  }
  if (text == "opencv_to_mujoco_y_back") {
    return {1.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0};
  }
  if (text == "ros_to_mujoco") {
    return {1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0};
  }
  auto values = parse_number_list(text);
  if (values.size() != 9) {
    die("basis must be one of identity/opencv_to_mujoco/ros_to_mujoco or 9 numbers");
  }
  std::array<double, 9> basis{};
  std::copy(values.begin(), values.end(), basis.begin());
  return basis;
}

std::array<double, 4> xyzw_to_wxyz(const std::array<double, 4> &q_xyzw) {
  double norm = std::sqrt(q_xyzw[0] * q_xyzw[0] + q_xyzw[1] * q_xyzw[1] +
                          q_xyzw[2] * q_xyzw[2] + q_xyzw[3] * q_xyzw[3]);
  if (norm < 1e-12) {
    return {1.0, 0.0, 0.0, 0.0};
  }
  return {q_xyzw[3] / norm, q_xyzw[0] / norm, q_xyzw[1] / norm, q_xyzw[2] / norm};
}

std::array<double, 9> quat_wxyz_to_mat(const std::array<double, 4> &q) {
  mjtNum quat[4] = {static_cast<mjtNum>(q[0]), static_cast<mjtNum>(q[1]),
                    static_cast<mjtNum>(q[2]), static_cast<mjtNum>(q[3])};
  mjtNum mat[9];
  mju_quat2Mat(mat, quat);
  return {mat[0], mat[1], mat[2],
          mat[3], mat[4], mat[5],
          mat[6], mat[7], mat[8]};
}

std::array<double, 4> mat_to_quat_wxyz(const std::array<double, 9> &mat) {
  mjtNum q[4];
  mjtNum m[9];
  for (int i = 0; i < 9; ++i) {
    m[i] = static_cast<mjtNum>(mat[i]);
  }
  mju_mat2Quat(q, m);
  double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (norm < 1e-12) {
    return {1.0, 0.0, 0.0, 0.0};
  }
  return {q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm};
}

std::array<double, 3> mat_vec(const std::array<double, 9> &m, const std::array<double, 3> &v) {
  return {
      m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
      m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
      m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  };
}

std::array<double, 9> mat_mul(const std::array<double, 9> &a, const std::array<double, 9> &b) {
  std::array<double, 9> out{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out[r * 3 + c] = 0.0;
      for (int k = 0; k < 3; ++k) {
        out[r * 3 + c] += a[r * 3 + k] * b[k * 3 + c];
      }
    }
  }
  return out;
}

std::vector<Pose> load_tum(const fs::path &path) {
  std::ifstream in(path);
  if (!in) {
    die("failed to open trajectory: " + path.string());
  }
  std::vector<Pose> poses;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream iss(line);
    Pose p;
    if (!(iss >> p.timestamp >> p.pos[0] >> p.pos[1] >> p.pos[2] >>
          p.quat_xyzw[0] >> p.quat_xyzw[1] >> p.quat_xyzw[2] >> p.quat_xyzw[3])) {
      continue;
    }
    poses.push_back(p);
  }
  if (poses.empty()) {
    die("no valid TUM poses found in " + path.string());
  }
  return poses;
}

std::vector<Pose> transform_poses(const std::vector<Pose> &input, const Args &args) {
  auto basis = parse_basis(args.basis);
  auto origin_offset = parse_vec3(args.origin_offset, "origin-offset");

  std::vector<Pose> out = input;
  std::array<double, 3> origin{0.0, 0.0, 0.0};
  if (args.origin == "first") {
    origin = {input.front().pos[0] * args.scale, input.front().pos[1] * args.scale, input.front().pos[2] * args.scale};
  } else if (args.origin != "zero") {
    die("origin must be first or zero");
  }

  for (auto &p : out) {
    std::array<double, 3> scaled = {p.pos[0] * args.scale, p.pos[1] * args.scale, p.pos[2] * args.scale};
    std::array<double, 3> centered = {scaled[0] - origin[0], scaled[1] - origin[1], scaled[2] - origin[2]};
    std::array<double, 3> mapped = mat_vec(basis, centered);
    p.pos = {mapped[0] + origin_offset[0], mapped[1] + origin_offset[1], mapped[2] + origin_offset[2] + args.z_offset};

    auto q_wxyz = xyzw_to_wxyz(p.quat_xyzw);
    auto rot = quat_wxyz_to_mat(q_wxyz);
    auto mapped_rot = mat_mul(basis, rot);
    auto q_new = mat_to_quat_wxyz(mapped_rot);
    p.quat_xyzw = {q_new[1], q_new[2], q_new[3], q_new[0]};
  }
  return out;
}

void set_mocap_pose(const mjModel *m, mjData *d, const std::string &body_name,
                    const std::array<double, 3> &pos,
                    const std::array<double, 4> &quat_xyzw) {
  int body_id = mj_name2id(m, mjOBJ_BODY, body_name.c_str());
  if (body_id < 0) {
    die("mocap body not found: " + body_name);
  }
  int mocap_id = static_cast<int>(m->body_mocapid[body_id]);
  if (mocap_id < 0) {
    die("body is not mocap: " + body_name);
  }
  d->mocap_pos[mocap_id * 3 + 0] = pos[0];
  d->mocap_pos[mocap_id * 3 + 1] = pos[1];
  d->mocap_pos[mocap_id * 3 + 2] = pos[2];
  auto q_wxyz = xyzw_to_wxyz(quat_xyzw);
  d->mocap_quat[mocap_id * 4 + 0] = q_wxyz[0];
  d->mocap_quat[mocap_id * 4 + 1] = q_wxyz[1];
  d->mocap_quat[mocap_id * 4 + 2] = q_wxyz[2];
  d->mocap_quat[mocap_id * 4 + 3] = q_wxyz[3];
}

std::string read_text(const fs::path &path) {
  std::ifstream in(path);
  if (!in) {
    die("failed to open xml: " + path.string());
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

PluginConfig parse_plugin_config(const std::string &xml_text, const std::string &plugin_name) {
  const std::string escaped_name = plugin_name;
  std::regex plugin_re("<plugin[^>]*name\\s*=\\s*\"" + escaped_name + "\"[\\s\\S]*?</plugin>");
  std::smatch plugin_match;
  if (!std::regex_search(xml_text, plugin_match, plugin_re)) {
    die("plugin block not found in xml: " + plugin_name);
  }

  PluginConfig cfg;
  std::regex config_re("<config[^>]*key\\s*=\\s*\"([^\"]+)\"[^>]*value\\s*=\\s*\"([^\"]*)\"[^>]*/?>");
  const std::string block = plugin_match.str();
  for (std::sregex_iterator it(block.begin(), block.end(), config_re), end; it != end; ++it) {
    cfg.kv[(*it)[1].str()] = (*it)[2].str();
  }
  return cfg;
}

std::string strip_draw_hip_point(const std::string &xml_text, const std::string &plugin_name) {
  std::regex plugin_re("(<plugin[^>]*name\\s*=\\s*\"" + plugin_name + "\"[\\s\\S]*?</plugin>)");
  std::smatch plugin_match;
  if (!std::regex_search(xml_text, plugin_match, plugin_re)) {
    return xml_text;
  }
  std::string block = plugin_match.str(1);
  std::regex draw_re("\\s*<config[^>]*key\\s*=\\s*\"draw_hip_point\"[^>]*/>\\s*");
  block = std::regex_replace(block, draw_re, "\n");
  return plugin_match.prefix().str() + block + plugin_match.suffix().str();
}

double cfg_double(const PluginConfig &cfg, const std::string &key, double default_value = 0.0) {
  auto it = cfg.kv.find(key);
  if (it == cfg.kv.end() || it->second.empty()) {
    return default_value;
  }
  return std::stod(it->second);
}

int cfg_int(const PluginConfig &cfg, const std::string &key, int default_value = 0) {
  auto it = cfg.kv.find(key);
  if (it == cfg.kv.end() || it->second.empty()) {
    return default_value;
  }
  return std::stoi(it->second);
}

std::pair<int, int> cfg_size(const PluginConfig &cfg) {
  auto it = cfg.kv.find("size");
  if (it == cfg.kv.end()) {
    die("size missing from plugin config");
  }
  auto values = parse_number_list(it->second);
  if (values.size() < 2) {
    die("size must contain width and height");
  }
  return {static_cast<int>(values[0]), static_cast<int>(values[1])};
}

std::array<double, 2> cfg_range(const PluginConfig &cfg) {
  auto it = cfg.kv.find("dis_range");
  if (it == cfg.kv.end()) {
    return {0.0, 6.0};
  }
  auto values = parse_number_list(it->second);
  if (values.size() < 2) {
    die("dis_range must contain two numbers");
  }
  return {values[0], values[1]};
}

RayCasterCameraCfg make_camera_cfg(const mjModel *m, mjData *d, const std::string &camera_name,
                                   const PluginConfig &cfg) {
  RayCasterCameraCfg out;
  out.m = m;
  out.d = d;
  out.cam_name = camera_name;
  out.focal_length = cfg_double(cfg, "focal_length", 1.0);
  out.horizontal_aperture = cfg_double(cfg, "horizontal_aperture", 2.0);
  out.vertical_aperture = cfg_double(cfg, "vertical_aperture", 0.0);
  auto size = cfg_size(cfg);
  out.h_ray_num = size.first;
  out.v_ray_num = size.second;
  auto range = cfg_range(cfg);
  out.dis_range = {range[0], range[1]};
  out.is_detect_parentbody = cfg_int(cfg, "detect_parentbody", 0) != 0;
  out.baseline = cfg_double(cfg, "baseline", 0.0);
  out.loss_angle = cfg_double(cfg, "lossangle", 0.0);
  out.min_energy = cfg_double(cfg, "min_energy", 0.0);
  return out;
}

void apply_noise(RayCasterCamera &camera, const PluginConfig &cfg) {
  const auto noise_type_it = cfg.kv.find("noise_type");
  if (noise_type_it == cfg.kv.end() || noise_type_it->second.empty()) {
    return;
  }
  const std::string noise_type = noise_type_it->second;
  const auto noise_cfg_it = cfg.kv.find("noise_cfg");
  const auto values = noise_cfg_it == cfg.kv.end() ? std::vector<double>{} : parse_number_list(noise_cfg_it->second);

  if (noise_type == "stereo_noise") {
    int power = values.size() >= 1 ? static_cast<int>(values[0]) : 1;
    unsigned int seed = values.size() >= 2 ? static_cast<unsigned int>(values[1]) : 0U;
    camera.setNoise(ray_noise::StereoNoise(power, seed));
    return;
  }
  if (noise_type == "noise1") {
    double low = values.size() >= 1 ? values[0] : 0.0;
    double high = values.size() >= 2 ? values[1] : 0.0;
    double zero_probability = values.size() >= 3 ? values[2] : 0.0;
    unsigned int seed = values.size() >= 4 ? static_cast<unsigned int>(values[3]) : 0U;
    camera.setNoise(ray_noise::RayNoise1(low, high, zero_probability, seed));
    return;
  }
  if (noise_type == "gaussian") {
    double mean = values.size() >= 1 ? values[0] : 0.0;
    double stddev = values.size() >= 2 ? values[1] : 0.0;
    unsigned int seed = values.size() >= 3 ? static_cast<unsigned int>(values[2]) : 0U;
    camera.setNoise(std_noise::GaussianNoise(mean, stddev, seed));
    return;
  }
  if (noise_type == "uniform") {
    double low = values.size() >= 1 ? values[0] : 0.0;
    double high = values.size() >= 2 ? values[1] : 0.0;
    unsigned int seed = values.size() >= 3 ? static_cast<unsigned int>(values[2]) : 0U;
    camera.setNoise(ray_noise::UniformNoise(low, high, seed));
    return;
  }
  die("unsupported noise_type for C++ collector: " + noise_type);
}

std::optional<std::pair<int, int>> read_first_real_rgb_size(const fs::path &dataset_dir) {
  std::ifstream in(dataset_dir / "associations.txt");
  if (!in) {
    return std::nullopt;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream iss(line);
    double rgb_t = 0.0;
    double depth_t = 0.0;
    std::string rgb_path;
    std::string depth_path;
    if (!(iss >> rgb_t >> rgb_path >> depth_t >> depth_path)) {
      continue;
    }
    cv::Mat image = cv::imread((dataset_dir / rgb_path).string(), cv::IMREAD_COLOR);
    if (!image.empty()) {
      return std::make_pair(image.cols, image.rows);
    }
  }
  return std::nullopt;
}

std::optional<std::pair<int, int>> parse_render_size(const std::string &text) {
  if (text == "auto") {
    return std::nullopt;
  }
  const auto x = text.find('x');
  if (x == std::string::npos) {
    die("render-size must be WIDTHxHEIGHT or auto");
  }
  return std::make_pair(std::stoi(text.substr(0, x)), std::stoi(text.substr(x + 1)));
}

void write_png_u16(const fs::path &path, const std::vector<float> &depth, int width, int height) {
  cv::Mat image(height, width, CV_16UC1, cv::Scalar(0));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float d = depth[y * width + x];
      if (std::isfinite(d) && d > 0.0f) {
        image.at<uint16_t>(y, x) = static_cast<uint16_t>(std::clamp(d * 1000.0f, 0.0f, 65535.0f));
      }
    }
  }
  cv::imwrite(path.string(), image);
}

std::vector<float> clip_depth_for_vis(const std::vector<float> &depth, float min_depth, float max_depth) {
  std::vector<float> out = depth;
  for (auto &v : out) {
    if (!std::isfinite(v) || v <= 0.0f) {
      v = 0.0f;
    } else if (v < min_depth) {
      v = 0.0f;
    } else if (v > max_depth) {
      v = max_depth;
    }
  }
  return out;
}

std::vector<float> zero_outside_clip(const std::vector<float> &depth, std::optional<double> min_depth,
                                     std::optional<double> max_depth) {
  if (!min_depth || !max_depth) {
    return depth;
  }
  std::vector<float> out = depth;
  for (auto &v : out) {
    if (!std::isfinite(v) || v <= 0.0f || v < *min_depth || v > *max_depth) {
      v = 0.0f;
    }
  }
  return out;
}

cv::Mat depth_to_vis(const std::vector<float> &depth, int width, int height, float min_depth, float max_depth) {
  auto clipped = clip_depth_for_vis(depth, min_depth, max_depth);
  cv::Mat image(height, width, CV_8UC1, cv::Scalar(0));
  const float denom = std::max(max_depth - min_depth, 1e-6f);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float d = clipped[y * width + x];
      if (d > 0.0f) {
        image.at<uint8_t>(y, x) = static_cast<uint8_t>(std::clamp((d - min_depth) / denom * 255.0f, 0.0f, 255.0f));
      }
    }
  }
  return image;
}

void write_npy_f32(const fs::path &path, const std::vector<float> &data, int height, int width) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    die("failed to open npy output: " + path.string());
  }
  const std::string magic = "\x93NUMPY";
  out.write(magic.data(), static_cast<std::streamsize>(magic.size()));
  const unsigned char version[2] = {1, 0};
  out.write(reinterpret_cast<const char *>(version), 2);

  std::ostringstream header_ss;
  header_ss << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << height << ", " << width << "), }";
  std::string header = header_ss.str();
  size_t preamble = magic.size() + 2 + 2;
  size_t padding = 16 - ((preamble + header.size() + 1) % 16);
  if (padding == 16) {
    padding = 0;
  }
  header.append(padding, ' ');
  header.push_back('\n');
  uint16_t header_len = static_cast<uint16_t>(header.size());
  out.write(reinterpret_cast<const char *>(&header_len), sizeof(header_len));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
}

std::string format_duration(double seconds) {
  const int total = std::max(0, static_cast<int>(std::round(seconds)));
  const int h = total / 3600;
  const int m = (total % 3600) / 60;
  const int s = total % 60;
  std::ostringstream oss;
  if (h > 0) {
    oss << h << ":" << std::setw(2) << std::setfill('0') << m << ":" << std::setw(2) << s;
  } else {
    oss << std::setw(2) << std::setfill('0') << m << ":" << std::setw(2) << s;
  }
  return oss.str();
}

void print_progress(int current, int total,
                    const std::chrono::steady_clock::time_point &started_at) {
  total = std::max(total, 1);
  current = std::min(current, total);
  const double frac = static_cast<double>(current) / static_cast<double>(total);
  const int width = 28;
  const int filled = std::min(width, static_cast<int>(std::round(frac * width)));
  const auto now = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration<double>(now - started_at).count();
  const double rate = elapsed > 1e-9 ? current / elapsed : 0.0;
  const double remaining = rate > 1e-9 ? (total - current) / rate : 0.0;
  std::ostringstream oss;
  oss << "\rProgress [";
  oss << std::string(filled, '#') << std::string(width - filled, '-');
  oss << "] " << current << "/" << total << " ";
  oss << std::fixed << std::setprecision(1) << std::setw(5) << frac * 100.0 << "%";
  oss << "  elapsed " << format_duration(elapsed) << "  eta " << format_duration(remaining);
  std::cout << oss.str() << std::flush;
}

class HiddenGlContext {
 public:
  HiddenGlContext(int width, int height) {
    if (!glfwInit()) {
      die("glfwInit failed");
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
    window_ = glfwCreateWindow(std::max(1, width), std::max(1, height), "mujoco_rgbd_collector", nullptr, nullptr);
    if (!window_) {
      glfwTerminate();
      die("glfwCreateWindow failed");
    }
    glfwMakeContextCurrent(window_);
  }

  ~HiddenGlContext() {
    if (window_) {
      glfwDestroyWindow(window_);
      glfwTerminate();
    }
  }

 private:
  GLFWwindow* window_ = nullptr;
};

class OffscreenRgbRenderer {
 public:
  OffscreenRgbRenderer(const mjModel* model, int camera_id, int width, int height)
      : model_(model), camera_id_(camera_id), width_(width), height_(height) {
    mjv_defaultScene(&scene_);
    mjr_defaultContext(&context_);
    mjv_defaultOption(&option_);
    mjv_defaultCamera(&camera_);
    mjv_makeScene(model_, &scene_, 2000);
    mjr_makeContext(model_, &context_, mjFONTSCALE_150);
    camera_.type = mjCAMERA_FIXED;
    camera_.fixedcamid = camera_id_;
  }

  ~OffscreenRgbRenderer() {
    mjr_freeContext(&context_);
    mjv_freeScene(&scene_);
  }

  cv::Mat render(mjData* data) {
    mjrRect rect{0, 0, width_, height_};
    std::vector<unsigned char> rgb(width_ * height_ * 3, 0);
    mjv_updateScene(model_, data, &option_, nullptr, &camera_, mjCAT_ALL, &scene_);
    mjr_setBuffer(mjFB_OFFSCREEN, &context_);
    mjr_render(rect, &scene_, &context_);
    mjr_readPixels(rgb.data(), nullptr, rect, &context_);
    cv::Mat image(height_, width_, CV_8UC3, rgb.data());
    cv::Mat flipped;
    cv::Mat bgr;
    cv::flip(image, flipped, 0);
    cv::cvtColor(flipped, bgr, cv::COLOR_RGB2BGR);
    return bgr;
  }

 private:
  const mjModel* model_ = nullptr;
  int camera_id_ = -1;
  int width_ = 0;
  int height_ = 0;
  mjvScene scene_;
  mjrContext context_;
  mjvOption option_;
  mjvCamera camera_;
};

cv::Mat render_rgb(OffscreenRgbRenderer& renderer, mjData* d) {
  return renderer.render(d);
}

std::string json_number(std::optional<double> value) {
  if (!value) {
    return "null";
  }
  std::ostringstream oss;
  oss << *value;
  return oss.str();
}

void write_collection_config(const fs::path &path, const Args &args, float vis_min_depth, float vis_max_depth,
                             int ray_w, int ray_h, int aux_w, int aux_h, int render_w, int render_h) {
  std::ofstream out(path);
  out << "{\n";
  out << "  \"depth_clip_min\": " << json_number(args.depth_clip_min) << ",\n";
  out << "  \"depth_clip_max\": " << json_number(args.depth_clip_max) << ",\n";
  out << "  \"min_depth\": " << vis_min_depth << ",\n";
  out << "  \"max_depth\": " << vis_max_depth << ",\n";
  out << "  \"raycaster_width\": " << ray_w << ",\n";
  out << "  \"raycaster_height\": " << ray_h << ",\n";
  out << "  \"secondary_raycaster_width\": " << aux_w << ",\n";
  out << "  \"secondary_raycaster_height\": " << aux_h << ",\n";
  out << "  \"render_width\": " << render_w << ",\n";
  out << "  \"render_height\": " << render_h << "\n";
  out << "}\n";
}

Args parse_args(int argc, char **argv) {
  Args args;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        die("missing value for " + name);
      }
      return argv[++i];
    };
    if (arg == "--xml") args.xml = need_value(arg);
    else if (arg == "--plugin") args.plugin = need_value(arg);
    else if (arg == "--body") args.body = need_value(arg);
    else if (arg == "--camera") args.camera = need_value(arg);
    else if (arg == "--sensor") args.primary_sensor = need_value(arg);
    else if (arg == "--secondary-sensor") args.secondary_sensor = need_value(arg);
    else if (arg == "--output") args.output = need_value(arg);
    else if (arg == "--every") args.every = std::stoi(need_value(arg));
    else if (arg == "--max-frames") args.max_frames = std::stoi(need_value(arg));
    else if (arg == "--render-size") args.render_size = need_value(arg);
    else if (arg == "--scale") args.scale = std::stod(need_value(arg));
    else if (arg == "--z-offset") args.z_offset = std::stod(need_value(arg));
    else if (arg == "--basis") args.basis = need_value(arg);
    else if (arg == "--origin") args.origin = need_value(arg);
    else if (arg == "--origin-offset") args.origin_offset = need_value(arg);
    else if (arg == "--min-depth") args.min_depth = std::stod(need_value(arg));
    else if (arg == "--max-depth") args.max_depth = std::stod(need_value(arg));
    else if (arg == "--depth-clip-min") args.depth_clip_min = std::stod(need_value(arg));
    else if (arg == "--depth-clip-max") args.depth_clip_max = std::stod(need_value(arg));
    else if (arg == "--no-render-rgb") args.render_rgb = false;
    else if (arg == "--render-rgb") args.render_rgb = true;
    else if (arg == "--no-progress") args.show_progress = false;
    else if (arg == "--viewer" || arg == "--fixed-view" || arg == "--viewer-fps" || arg == "--render-depth" ||
             arg == "--no-render-depth" || arg == "--render-depth-noise-std" || arg == "--render-depth-noise-seed" ||
             arg == "--render-depth-clip-min" || arg == "--render-depth-clip-max" || arg == "--depth-block") {
      if (arg == "--viewer-fps" || arg == "--render-depth-noise-std" || arg == "--render-depth-noise-seed" ||
          arg == "--render-depth-clip-min" || arg == "--render-depth-clip-max" || arg == "--depth-block") {
        ++i;
      }
      // accepted for shell compatibility, ignored in the C++ collector
    } else if (!arg.empty() && arg[0] == '-') {
      die("unknown argument: " + arg);
    } else {
      positional.push_back(arg);
    }
  }

  if (positional.empty()) {
    die("trajectory path is required");
  }
  args.trajectory = positional[0];
  if (args.output.empty()) {
    die("--output is required");
  }
  if ((args.depth_clip_min.has_value()) != (args.depth_clip_max.has_value())) {
    die("--depth-clip-min and --depth-clip-max must be provided together");
  }
  return args;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    Args args = parse_args(argc, argv);
    args.trajectory = fs::absolute(args.trajectory);
    args.output = fs::absolute(args.output);
    args.xml = fs::absolute(args.xml);
    args.plugin = fs::absolute(args.plugin);

    if (!fs::exists(args.plugin)) {
      die("plugin library not found: " + args.plugin.string());
    }
    mj_loadPluginLibrary(args.plugin.string().c_str());

    const std::string xml_text = read_text(args.xml);
    const PluginConfig primary_cfg = parse_plugin_config(xml_text, args.primary_sensor);
    const PluginConfig secondary_cfg = parse_plugin_config(xml_text, args.secondary_sensor);

    const std::string render_xml_text = strip_draw_hip_point(xml_text, args.primary_sensor);
    const fs::path render_xml_path = args.xml.parent_path() / (args.xml.stem().string() + "_render_clean_cpp.xml");
    {
      std::ofstream out(render_xml_path);
      out << render_xml_text;
    }

    mjModel *model = mj_loadXML(args.xml.string().c_str(), nullptr, nullptr, 0);
    if (!model) {
      die("failed to load MuJoCo XML: " + args.xml.string());
    }
    mjData *data = mj_makeData(model);

    mjModel *render_model = mj_loadXML(render_xml_path.string().c_str(), nullptr, nullptr, 0);
    if (!render_model) {
      die("failed to load clean render XML: " + render_xml_path.string());
    }
    mjData *render_data = mj_makeData(render_model);

    const auto poses = transform_poses(load_tum(args.trajectory), args);
    set_mocap_pose(model, data, args.body, poses.front().pos, poses.front().quat_xyzw);
    mj_forward(model, data);
    set_mocap_pose(render_model, render_data, args.body, poses.front().pos, poses.front().quat_xyzw);
    mj_forward(render_model, render_data);

    RayCasterCamera primary_camera(make_camera_cfg(model, data, args.camera, primary_cfg));
    apply_noise(primary_camera, primary_cfg);
    RayCasterCamera secondary_camera(make_camera_cfg(model, data, args.camera, secondary_cfg));
    apply_noise(secondary_camera, secondary_cfg);

    const auto primary_size = cfg_size(primary_cfg);
    const auto secondary_size = cfg_size(secondary_cfg);
    const auto sensor_range = cfg_range(primary_cfg);
    const float vis_min_depth = static_cast<float>(args.min_depth.value_or(sensor_range[0]));
    const float vis_max_depth = static_cast<float>(args.max_depth.value_or(sensor_range[1]));

    auto render_size = parse_render_size(args.render_size);
    if (!render_size.has_value()) {
      render_size = read_first_real_rgb_size(args.trajectory.parent_path());
    }
    if (!render_size.has_value()) {
      const int camera_id = mj_name2id(render_model, mjOBJ_CAMERA, args.camera.c_str());
      if (camera_id < 0) {
        die("camera not found: " + args.camera);
      }
      render_size = std::make_pair(
          render_model->cam_resolution[camera_id * 2 + 0],
          render_model->cam_resolution[camera_id * 2 + 1]);
    }
    const int render_w = render_size->first;
    const int render_h = render_size->second;
    const int render_camera_id = mj_name2id(render_model, mjOBJ_CAMERA, args.camera.c_str());
    if (render_camera_id < 0) {
      die("render camera not found: " + args.camera);
    }
    std::optional<HiddenGlContext> gl_context;
    std::optional<OffscreenRgbRenderer> rgb_renderer;
    if (args.render_rgb) {
      gl_context.emplace(render_w, render_h);
      rgb_renderer.emplace(render_model, render_camera_id, render_w, render_h);
    }

    const fs::path primary_npy_dir = args.output / "sim_depth_float32_m";
    const fs::path primary_png_dir = args.output / "sim_depth_uint16_mm";
    const fs::path primary_vis_dir = args.output / "sim_depth_vis";
    const fs::path secondary_npy_dir = args.output / "secondary_depth_float32_m";
    const fs::path secondary_png_dir = args.output / "secondary_depth_uint16_mm";
    const fs::path secondary_vis_dir = args.output / "secondary_depth_vis";
    const fs::path rgb_dir = args.output / "sim_rgb";
    fs::create_directories(primary_npy_dir);
    fs::create_directories(primary_png_dir);
    fs::create_directories(primary_vis_dir);
    fs::create_directories(secondary_npy_dir);
    fs::create_directories(secondary_png_dir);
    fs::create_directories(secondary_vis_dir);
    if (args.render_rgb) {
      fs::create_directories(rgb_dir);
    }

    const fs::path meta_path = args.output / "frames.txt";
    const fs::path config_path = args.output / "collection_config.json";
    write_collection_config(config_path, args, vis_min_depth, vis_max_depth,
                            primary_size.first, primary_size.second,
                            secondary_size.first, secondary_size.second,
                            render_w, render_h);

    std::ofstream meta(meta_path);
    meta << "# out_index traj_index timestamp ray_depth_npy ray_depth_png ray_depth_vis sim_rgb secondary_depth_npy secondary_depth_png secondary_depth_vis\n";

    const int step = std::max(1, args.every);
    int total_frames = static_cast<int>((poses.size() + step - 1) / step);
    if (args.max_frames > 0) {
      total_frames = std::min(total_frames, args.max_frames);
    }
    int collected = 0;
    const auto started_at = std::chrono::steady_clock::now();

    for (int traj_idx = 0; traj_idx < static_cast<int>(poses.size()); traj_idx += step) {
      if (args.max_frames > 0 && collected >= args.max_frames) {
        break;
      }

      set_mocap_pose(model, data, args.body, poses[traj_idx].pos, poses[traj_idx].quat_xyzw);
      mj_forward(model, data);
      set_mocap_pose(render_model, render_data, args.body, poses[traj_idx].pos, poses[traj_idx].quat_xyzw);
      mj_forward(render_model, render_data);

      primary_camera.compute_distance();
      secondary_camera.compute_distance();

      auto primary_raw = primary_camera.get_distance_to_image_plane_vec(true, false);
      auto secondary_raw = secondary_camera.get_distance_to_image_plane_vec(true, false);
      std::vector<float> primary(primary_raw.begin(), primary_raw.end());
      std::vector<float> secondary(secondary_raw.begin(), secondary_raw.end());
      primary = zero_outside_clip(primary, args.depth_clip_min, args.depth_clip_max);
      secondary = zero_outside_clip(secondary, args.depth_clip_min, args.depth_clip_max);
      const auto primary_vis_source = clip_depth_for_vis(primary, vis_min_depth, vis_max_depth);
      const auto secondary_vis_source = clip_depth_for_vis(secondary, vis_min_depth, vis_max_depth);

      const std::string stem = [&] {
        std::ostringstream oss;
        oss << std::setw(6) << std::setfill('0') << collected;
        return oss.str();
      }();

      write_npy_f32(primary_npy_dir / (stem + ".npy"), primary, primary_size.second, primary_size.first);
      write_png_u16(primary_png_dir / (stem + ".png"), primary, primary_size.first, primary_size.second);
      cv::imwrite((primary_vis_dir / (stem + ".png")).string(),
                  depth_to_vis(primary_vis_source, primary_size.first, primary_size.second, vis_min_depth, vis_max_depth));

      write_npy_f32(secondary_npy_dir / (stem + ".npy"), secondary, secondary_size.second, secondary_size.first);
      write_png_u16(secondary_png_dir / (stem + ".png"), secondary, secondary_size.first, secondary_size.second);
      cv::imwrite((secondary_vis_dir / (stem + ".png")).string(),
                  depth_to_vis(secondary_vis_source, secondary_size.first, secondary_size.second, vis_min_depth, vis_max_depth));

      std::string rgb_name = "-";
      if (args.render_rgb) {
        cv::Mat sim_rgb = render_rgb(*rgb_renderer, render_data);
        rgb_name = "sim_rgb/" + stem + ".png";
        cv::imwrite((rgb_dir / (stem + ".png")).string(), sim_rgb);
      }

      meta << collected << " " << traj_idx << " " << std::fixed << std::setprecision(9) << poses[traj_idx].timestamp
           << " sim_depth_float32_m/" << stem << ".npy"
           << " sim_depth_uint16_mm/" << stem << ".png"
           << " sim_depth_vis/" << stem << ".png"
           << " " << rgb_name
           << " secondary_depth_float32_m/" << stem << ".npy"
           << " secondary_depth_uint16_mm/" << stem << ".png"
           << " secondary_depth_vis/" << stem << ".png\n";

      ++collected;
      if (args.show_progress) {
        print_progress(collected, total_frames, started_at);
      }
    }

    if (args.show_progress && collected > 0) {
      std::cout << "\n";
    }

    std::cout << "Collected " << collected << " depth frames\n";
    std::cout << "Primary ray caster size: " << primary_size.first << "x" << primary_size.second << "\n";
    std::cout << "Secondary ray caster size: " << secondary_size.first << "x" << secondary_size.second << "\n";
    std::cout << "Depth visualization range: " << vis_min_depth << ".." << vis_max_depth << " m\n";
    if (args.depth_clip_min && args.depth_clip_max) {
      std::cout << "Shared depth zero-outside clip: " << *args.depth_clip_min << ".." << *args.depth_clip_max << " m\n";
    }
    if (args.render_rgb) {
      std::cout << "Sim RGB size: " << render_w << "x" << render_h << "\n";
    }
    std::cout << "Output: " << args.output << "\n";

    mj_deleteData(render_data);
    mj_deleteModel(render_model);
    mj_deleteData(data);
    mj_deleteModel(model);
    std::error_code ec;
    fs::remove(render_xml_path, ec);
    return 0;
  } catch (const std::exception &exc) {
    std::cerr << "Error: " << exc.what() << "\n";
    return 1;
  }
}
