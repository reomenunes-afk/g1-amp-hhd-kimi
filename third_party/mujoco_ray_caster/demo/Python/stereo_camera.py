import mujoco
import mujoco_viewer
import cv2
import numpy as np

# 加载插件和模型
mujoco.mj_loadPluginLibrary('../../lib/libsensor_raycaster.so')
m = mujoco.MjModel.from_xml_path("../../model/ray_caster4.xml")
d = mujoco.MjData(m)

def get_ray_caster_info(model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str):
    data_ps = []
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id == -1:
        print("Sensor not found")
        return 0, 0, data_ps
    sensor_plugin_id = model.sensor_plugin[sensor_id]
    state_idx = model.plugin_stateadr[sensor_plugin_id]
    state_num = model.plugin_statenum[sensor_plugin_id]
    for i in range(state_idx + 2, state_idx + state_num, 2):
        if i + 1 < len(data.plugin_state):
            data_ps.append((int(data.plugin_state[i]), int(data.plugin_state[i + 1])))
    
    # 安全读取 h_ray_num 和 v_ray_num
    h_ray_num = int(data.plugin_state[state_idx]) if state_idx < len(data.plugin_state) else 0
    v_ray_num = int(data.plugin_state[state_idx + 1]) if state_idx + 1 < len(data.plugin_state) else 0
    return h_ray_num, v_ray_num, data_ps

sensor_name = "raycastercamera"
h_rays, v_rays, pairs = get_ray_caster_info(m, d, sensor_name)
print(f"h_rays: {h_rays}")
print(f"v_rays: {v_rays}")
print(f"[data_point,data_size]: {pairs}")

viewer = mujoco_viewer.MujocoViewer(m, d, width=1920, height=1080)

# ================= 配置区域 =================
# 在这里设定你想要的固定归一化范围
# 例如：深度图可能是 0.0 到 5.0，能量图可能是 0.0 到 1.0
GLOBAL_VMIN = 0.01  
GLOBAL_VMAX = 3.0  
# ===========================================

def visualize_block(block_data, length, idx, h_rays, v_rays, win_prefix="block", vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX):
    """
    block_data: 1D numpy array
    length:     数据长度
    idx:        第几个 block
    vmin/vmax:  固定的归一化范围
    """
    hw = h_rays * v_rays
    img = None

    # 1. Reshape 逻辑保持不变
    if length == hw:
        img = block_data.reshape(v_rays, h_rays)
    elif length == hw * 3:
        img = block_data.reshape(v_rays, h_rays, 3)
        # 如果是多通道，通常只对亮度归一化，这里简化处理：如果是RGB通常不需要归一化到0-1再转255，除非是浮点RGB
        # 假设这里是单通道数据显示，如果是RGB且范围已知(0-1)，可以直接乘255
        if img.dtype == np.float32:
             img = img[:, :, 0] # 暂时只取第一通道演示单通道归一化逻辑
    elif length % hw == 0 and length // hw <= 4:
        c = length // hw
        img = block_data.reshape(v_rays, h_rays, c)
        if c != 3:
            img = img[:, :, 0]
    else:
        side = int(np.ceil(np.sqrt(length)))
        tmp = np.zeros((side * side,), dtype=np.float32)
        tmp[:length] = block_data
        img = tmp.reshape(side, side)

    img = img.astype(np.float32)

    # ================= 修改核心：固定范围归一化 =================
    # 1. 截断数据：将超出 [vmin, vmax] 的值强制拉到边界
    img_clipped = np.clip(img, vmin, vmax)
    
    # 2. 防止除以零：如果范围无效，全黑
    range_val = vmax - vmin
    if range_val > 0:
        img_norm = (img_clipped - vmin) / range_val
    else:
        img_norm = np.zeros_like(img_clipped)
    
    # 3. 转换到 0-255
    img_uint8 = (img_norm * 255).astype(np.uint8)
    # =========================================================

    # 放大显示
    scale = 4
    h, w = img_uint8.shape[:2]
    img_show = cv2.resize(
        img_uint8, (w * scale, h * scale),
        interpolation=cv2.INTER_NEAREST
    )

    # 在图像上显示当前的固定范围信息，方便调试
    # cv2.putText(img_show, f"Range: [{vmin:.2f}, {vmax:.2f}]", (10, 20), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    win_name = f"{win_prefix}_{idx}_len{length}"
    cv2.imshow(win_name, img_show)

while viewer.is_alive:
    mujoco.mj_step(m, d)
    viewer.render()

    full_data = d.sensor(sensor_name).data

    for i, (start, size) in enumerate(pairs):
        block = np.array(full_data[start:start+size], dtype=np.float32)
        
        # 【可选】如果你不同的 block 需要不同的范围，可以在这里覆盖 vmin/vmax
        # 例如：如果是深度数据 (假设是第0个block)，范围设大一点
        # cur_vmin, cur_vmax = 0.0, 10.0 if i == 0 else (0.0, 1.0)
        
        visualize_block(block, size, i, h_rays, v_rays, win_prefix=sensor_name)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
viewer.close()