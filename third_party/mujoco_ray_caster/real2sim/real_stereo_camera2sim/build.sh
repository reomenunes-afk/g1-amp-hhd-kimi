#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAYCASTER_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

MJ_CONDA_ENV="${MJ_CONDA_ENV:-mj}"
ORB_SLAM3_REPO="${ORB_SLAM3_REPO:-https://github.com/UZ-SLAMLab/ORB_SLAM3.git}"
ORB_SLAM3_REF="${ORB_SLAM3_REF:-}"
ORB_SLAM3_ROOT="${ORB_SLAM3_ROOT:-$PROJECT_DIR/third_party/ORB_SLAM3}"

YES=0
SKIP_DEPS=0
SKIP_ORB=0
SKIP_RAYCASTER=0

usage() {
  cat <<'EOF'
Usage: ./build.sh [options]

Build real_stereo_camera2sim inside mujoco_ray_caster.

Options:
  --yes              Answer yes to prompts.
  --skip-deps        Do not ask to check environment dependencies.
  --skip-orbslam     Do not clone/build ORB-SLAM3.
  --skip-raycaster   Do not build libsensor_raycaster.so.
  -h, --help         Show this help.

Environment:
  MJ_CONDA_ENV       Conda environment containing mujoco, default: mj
  ORB_SLAM3_ROOT     ORB-SLAM3 source/build directory, default: third_party/ORB_SLAM3
  ORB_SLAM3_REPO     ORB-SLAM3 git URL, default: official ORB_SLAM3 repo
  ORB_SLAM3_REF      Optional branch/tag/commit to checkout after clone
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) YES=1 ;;
    --skip-deps) SKIP_DEPS=1 ;;
    --skip-orbslam) SKIP_ORB=1 ;;
    --skip-raycaster) SKIP_RAYCASTER=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

ask_yes_no() {
  local prompt="$1"
  if [[ "$YES" == "1" ]]; then
    return 0
  fi
  read -r -p "$prompt [y/N] " answer
  [[ "$answer" == "y" || "$answer" == "Y" || "$answer" == "yes" || "$answer" == "YES" ]]
}

check_command() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    echo "  ok: $name"
  else
    echo "  missing: $name"
    return 1
  fi
}

check_dependencies() {
  echo "Checking build/runtime dependencies..."
  local missing=0

  for cmd in git cmake g++ make pkg-config python3 conda; do
    check_command "$cmd" || missing=1
  done

  if pkg-config --exists glfw3; then
    echo "  ok: glfw3"
  else
    echo "  missing: glfw3 development package"
    missing=1
  fi

  if pkg-config --exists opencv4; then
    echo "  ok: opencv4"
  else
    echo "  missing: opencv4 development package"
    missing=1
  fi

  if conda run --no-capture-output -n "$MJ_CONDA_ENV" python - <<'PY' >/dev/null 2>&1
import mujoco, cv2, numpy
print("ok")
PY
  then
    echo "  ok: conda env '$MJ_CONDA_ENV' has mujoco/cv2/numpy"
  else
    echo "  missing: conda env '$MJ_CONDA_ENV' with mujoco/cv2/numpy"
    missing=1
  fi

  if [[ -f /opt/ros/humble/setup.bash ]]; then
    echo "  ok: ROS2 Humble setup.bash"
  else
    echo "  note: ROS2 Humble not found at /opt/ros/humble/setup.bash"
  fi

  if [[ "$missing" == "1" ]]; then
    cat <<'EOF'

Some dependencies are missing. Common Ubuntu packages:
  sudo apt-get update
  sudo apt-get install -y build-essential cmake git pkg-config \
    libopencv-dev libglfw3-dev libeigen3-dev libboost-all-dev \
    libssl-dev libpython3-dev

ORB-SLAM3 also needs Pangolin. If your machine already has Pangolin in a
custom prefix, make sure LD_LIBRARY_PATH can find it at runtime.
EOF
    return 1
  fi
}

patch_orbslam3_rgbd_tum() {
  local orb_root="$1"
  python3 - "$orb_root" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / "Examples" / "RGB-D" / "rgbd_tum.cc"
text = path.read_text()

if "Tracking lost at frame" in text:
    print("ORB-SLAM3 rgbd_tum patch already present.")
    raise SystemExit(0)

if "#include<Tracking.h>" not in text:
    text = text.replace("#include<System.h>\n", "#include<System.h>\n#include<Tracking.h>\n")

text = text.replace(
    "    // Main loop\n    cv::Mat imRGB, imD;\n    for(int ni=0; ni<nImages; ni++)\n",
    "    // Main loop\n    cv::Mat imRGB, imD;\n    int nProcessed = 0;\n    for(int ni=0; ni<nImages; ni++)\n",
)
text = text.replace(
    "        SLAM.TrackRGBD(imRGB,imD,tframe);\n",
    "        SLAM.TrackRGBD(imRGB,imD,tframe);\n        nProcessed = ni + 1;\n",
)
text = text.replace(
    "        // Wait to load the next frame\n",
    """        const int trackingState = SLAM.GetTrackingState();
        if(trackingState==ORB_SLAM3::Tracking::RECENTLY_LOST ||
           trackingState==ORB_SLAM3::Tracking::LOST)
        {
            cout << "Tracking lost at frame " << ni << " / " << nImages
                 << ". Stopping sequence early and saving trajectory up to this point." << endl;
            break;
        }

        // Wait to load the next frame
""",
)
text = text.replace(
    "    // Tracking time statistics\n    sort(vTimesTrack.begin(),vTimesTrack.end());\n",
    "    // Tracking time statistics\n    if(nProcessed < 1)\n        nProcessed = nImages;\n    vTimesTrack.resize(nProcessed);\n    sort(vTimesTrack.begin(),vTimesTrack.end());\n",
)
text = text.replace("    for(int ni=0; ni<nImages; ni++)\n    {\n        totaltime+=vTimesTrack[ni];\n    }\n", "    for(int ni=0; ni<nProcessed; ni++)\n    {\n        totaltime+=vTimesTrack[ni];\n    }\n")
text = text.replace('    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;\n', '    cout << "processed frames: " << nProcessed << " / " << nImages << endl;\n    cout << "median tracking time: " << vTimesTrack[nProcessed/2] << endl;\n')
text = text.replace('    cout << "mean tracking time: " << totaltime/nImages << endl;\n', '    cout << "mean tracking time: " << totaltime/nProcessed << endl;\n')

path.write_text(text)
print("Patched ORB-SLAM3 rgbd_tum.cc for fail-fast RGB-D trajectory export.")
PY
}

ensure_orbslam3() {
  if [[ "$SKIP_ORB" == "1" ]]; then
    echo "Skipping ORB-SLAM3 build."
    return
  fi

  if [[ ! -d "$ORB_SLAM3_ROOT" ]]; then
    mkdir -p "$(dirname "$ORB_SLAM3_ROOT")"
    echo "Cloning ORB-SLAM3 into: $ORB_SLAM3_ROOT"
    git clone --recursive "$ORB_SLAM3_REPO" "$ORB_SLAM3_ROOT"
  fi

  if [[ -n "$ORB_SLAM3_REF" ]]; then
    git -C "$ORB_SLAM3_ROOT" fetch --all --tags
    git -C "$ORB_SLAM3_ROOT" checkout "$ORB_SLAM3_REF"
  fi

  patch_orbslam3_rgbd_tum "$ORB_SLAM3_ROOT"

  echo "Building ORB-SLAM3 at: $ORB_SLAM3_ROOT"
  chmod +x "$ORB_SLAM3_ROOT/build.sh"
  (cd "$ORB_SLAM3_ROOT" && ./build.sh)

  if [[ ! -x "$ORB_SLAM3_ROOT/Examples/RGB-D/rgbd_tum" ]]; then
    echo "ORB-SLAM3 build did not produce Examples/RGB-D/rgbd_tum" >&2
    exit 1
  fi
  if [[ ! -f "$ORB_SLAM3_ROOT/Vocabulary/ORBvoc.txt" ]]; then
    echo "ORB-SLAM3 vocabulary not found: $ORB_SLAM3_ROOT/Vocabulary/ORBvoc.txt" >&2
    exit 1
  fi
}

if [[ "$SKIP_DEPS" != "1" ]]; then
  if ask_yes_no "Check environment dependencies before building?"; then
    check_dependencies || true
  else
    echo "Skipping dependency check."
  fi
fi

if [[ "$SKIP_RAYCASTER" != "1" ]]; then
  echo "Building mujoco_ray_caster plugin..."
  "$PROJECT_DIR/build_mujoco_raycaster_plugin.sh"
fi

ensure_orbslam3

echo "Building real_stereo_camera2sim collector..."
"$PROJECT_DIR/build_mujoco_rgbd_collector.sh"

cat <<EOF

Build finished.

Ray caster plugin:
  $RAYCASTER_ROOT/lib/libsensor_raycaster.so

ORB-SLAM3 root:
  $ORB_SLAM3_ROOT

Collector:
  $PROJECT_DIR/bin/mujoco_rgbd_collector
EOF
