#!/usr/bin/env python3
"""Adapt a MuJoCo MJCF generated from the training URDF for the Bitbot sim2sim pipeline.

MuJoCo's URDF importer drops the root link and turns its children into worldbody roots.
This script rebuilds a floating torso_link root and re-attaches the legs/arms so the
model matches the training topology and base frame.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import math
import sys


def rpy_to_quat(rpy: str) -> str:
    """Convert 'roll pitch yaw' (XYZ Euler) to MuJoCo quat 'w x y z'."""
    vals = [float(v) for v in rpy.split()]
    if len(vals) != 3:
        return '1 0 0 0'
    roll, pitch, yaw = vals
    if abs(roll) < 1e-9 and abs(pitch) < 1e-9 and abs(yaw) < 1e-9:
        return None
    # half angles
    cr, sr = (0.0, 0.0) if abs(roll) < 1e-9 else (math.cos(roll * 0.5), math.sin(roll * 0.5))
    cp, sp = (0.0, 0.0) if abs(pitch) < 1e-9 else (math.cos(pitch * 0.5), math.sin(pitch * 0.5))
    cy, sy = (0.0, 0.0) if abs(yaw) < 1e-9 else (math.cos(yaw * 0.5), math.sin(yaw * 0.5))
    w = cr * cp * cy - sr * sp * sy
    x = sr * cp * cy + cr * sp * sy
    y = cr * sp * cy - sr * cp * sy
    z = cr * cp * sy + sr * sp * cy
    return f'{w:.9f} {x:.9f} {y:.9f} {z:.9f}'


def get_torso_inertial(urdf_path: Path):
    """Read the torso_link inertial from the source URDF."""
    urdf = ET.parse(urdf_path).getroot()
    link = urdf.find('.//link[@name="torso_link"]')
    if link is None:
        raise RuntimeError('torso_link not found in URDF')
    inertial = link.find('inertial')
    origin = inertial.find('origin')
    pos = origin.get('xyz') if origin is not None else '0 0 0'
    rpy = origin.get('rpy') if origin is not None else '0 0 0'
    mass = inertial.find('mass').get('value')
    inertia = inertial.find('inertia')
    ixx = inertia.get('ixx'); iyy = inertia.get('iyy'); izz = inertia.get('izz')
    ixy = inertia.get('ixy'); ixz = inertia.get('ixz'); iyz = inertia.get('iyz')
    return {
        'pos': pos,
        'quat': rpy_to_quat(rpy),
        'mass': mass,
        'fullinertia': f'{ixx} {iyy} {izz} {ixy} {ixz} {iyz}',
    }


# The URDF references a few meshes with revision suffixes that do not exist in
# the local meshes/ directory; map them to the available file names.
MESH_FILENAME_MAP = {
    'torso_link_rev_1_0.STL': 'torso_link.STL',
    'waist_yaw_link_rev_1_0.STL': 'waist_yaw_link.STL',
    'waist_roll_link_rev_1_0.STL': 'waist_roll_link.STL',
}


def collect_visual_meshes(urdf_path: Path):
    """Parse URDF <visual> tags and return {link_name: [(mesh_name, pos, quat), ...]}."""
    if urdf_path is None or not urdf_path.exists():
        return {}
    urdf = ET.parse(urdf_path).getroot()
    meshes = {}
    for link in urdf.findall('link'):
        name = link.get('name')
        items = []
        for visual in link.findall('visual'):
            mesh_el = visual.find('.//mesh')
            if mesh_el is None:
                continue
            fname = mesh_el.get('filename')
            if fname is None:
                continue
            fname = fname.replace('\\', '/').split('/')[-1]
            fname = MESH_FILENAME_MAP.get(fname, fname)

            origin = visual.find('origin')
            pos = origin.get('xyz') if origin is not None else '0 0 0'
            rpy = origin.get('rpy') if origin is not None else '0 0 0'
            quat = rpy_to_quat(rpy)

            mesh_name = Path(fname).stem
            items.append((mesh_name, pos, quat, fname))
        if items:
            meshes[name] = items
    return meshes


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    urdf = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    tree = ET.parse(src)
    root = tree.getroot()

    # 1. Compiler
    compiler = root.find('compiler')
    if compiler is None:
        compiler = ET.Element('compiler')
        root.insert(0, compiler)
    compiler.set('angle', 'radian')
    compiler.set('autolimits', 'true')
    compiler.set('meshdir', 'meshes')

    # 2. Defaults: per-joint armature/damping classes matching InstinctLab training.
    default = ET.Element('default')
    root.insert(0, default)

    classes = {
        'motor_7520_14': {'armature': '0.01017752'},
        'motor_7520_22': {'armature': '0.025101925'},
        'motor_5020_x2': {'armature': '0.00721945'},
        'motor_5020': {'armature': '0.003609725'},
        'motor_4010': {'armature': '0.00425'},
    }
    for name, attribs in classes.items():
        cls = ET.SubElement(default, 'default')
        cls.set('class', name)
        joint = ET.SubElement(cls, 'joint')
        joint.set('damping', '0.0')
        joint.set('frictionloss', '0.0')
        joint.set('armature', attribs['armature'])

    joint_class_map = {
        'left_hip_pitch_joint': 'motor_7520_14',
        'left_hip_roll_joint': 'motor_7520_22',
        'left_hip_yaw_joint': 'motor_7520_14',
        'left_knee_joint': 'motor_7520_22',
        'left_ankle_pitch_joint': 'motor_5020_x2',
        'left_ankle_roll_joint': 'motor_5020_x2',
        'right_hip_pitch_joint': 'motor_7520_14',
        'right_hip_roll_joint': 'motor_7520_22',
        'right_hip_yaw_joint': 'motor_7520_14',
        'right_knee_joint': 'motor_7520_22',
        'right_ankle_pitch_joint': 'motor_5020_x2',
        'right_ankle_roll_joint': 'motor_5020_x2',
        'waist_yaw_joint': 'motor_7520_14',
        'waist_roll_joint': 'motor_5020_x2',
        'waist_pitch_joint': 'motor_5020_x2',
        'left_shoulder_pitch_joint': 'motor_5020',
        'left_shoulder_roll_joint': 'motor_5020',
        'left_shoulder_yaw_joint': 'motor_5020',
        'left_elbow_joint': 'motor_5020',
        'left_wrist_roll_joint': 'motor_5020',
        'left_wrist_pitch_joint': 'motor_4010',
        'left_wrist_yaw_joint': 'motor_4010',
        'right_shoulder_pitch_joint': 'motor_5020',
        'right_shoulder_roll_joint': 'motor_5020',
        'right_shoulder_yaw_joint': 'motor_5020',
        'right_elbow_joint': 'motor_5020',
        'right_wrist_roll_joint': 'motor_5020',
        'right_wrist_pitch_joint': 'motor_4010',
        'right_wrist_yaw_joint': 'motor_4010',
    }

    for body in root.iter('body'):
        for joint in body.findall('joint'):
            name = joint.get('name')
            if name in joint_class_map:
                joint.set('class', joint_class_map[name])

    # 3. Rebuild a floating torso_link root.
    worldbody = root.find('worldbody')

    # Collect leading geoms that belong to the dropped torso_link.
    torso_geoms = []
    while True:
        geom = worldbody.find('geom')
        if geom is None:
            break
        torso_geoms.append(geom)
        worldbody.remove(geom)

    # All remaining top-level bodies are children that were connected to torso_link.
    child_bodies = list(worldbody.findall('body'))
    if not child_bodies:
        raise RuntimeError('No child bodies found under worldbody')

    torso = ET.Element('body')
    torso.set('name', 'torso_link')
    torso.set('pos', '0 0 0.9')

    # Inertial from URDF if available, otherwise use known values.
    if urdf is not None and urdf.exists():
        inertial_info = get_torso_inertial(urdf)
        torso_inertial = ET.SubElement(torso, 'inertial')
        torso_inertial.set('pos', inertial_info['pos'])
        if inertial_info['quat'] is not None:
            torso_inertial.set('quat', inertial_info['quat'])
        torso_inertial.set('mass', inertial_info['mass'])
        torso_inertial.set('fullinertia', inertial_info['fullinertia'])
    else:
        torso_inertial = ET.SubElement(torso, 'inertial')
        torso_inertial.set('pos', '0.000931 0.000346 0.15082')
        torso_inertial.set('mass', '6.78')
        torso_inertial.set('fullinertia', '0.05905 3.3302e-05 -0.0017715 0.047014 -2.2399e-05 0.025652')

    for geom in torso_geoms:
        torso.append(geom)

    free_joint = ET.SubElement(torso, 'joint')
    free_joint.set('name', 'floating_base_joint')
    free_joint.set('type', 'free')
    free_joint.set('limited', 'false')

    # Attach the child bodies directly to torso_link.  Each child already
    # contains the joint that connects it to its parent (torso_link).
    for child in child_bodies:
        torso.append(child)

    # IMU and depth camera are attached to the torso (training base frame).
    imu_site = ET.Element('site')
    imu_site.set('name', 'imu')
    imu_site.set('size', '0.01')
    imu_site.set('pos', '0 0 0')
    torso.insert(0, imu_site)

    cam = ET.Element('camera')
    cam.set('name', 'depth_cam')
    cam.set('pos', '0.0488 0.01 0.4378')
    # Empirically tuned camera orientation that keeps the policy upright in MuJoCo.
    # The training-derived quaternion produces a horizontally-looking camera that
    # causes the robot to crouch in an empty scene, so we keep a slight downward
    # pitch that lets the depth sensor see the floor.
    cam.set('xyaxes', '-0.00355 -0.99996 -0.00797 0.74302 -0.00797 0.66909')
    cam.set('fovy', '58.29')
    cam.set('mode', 'fixed')
    torso.insert(1, cam)

    # Replace worldbody children with the new torso.
    for child in list(worldbody):
        worldbody.remove(child)
    worldbody.insert(0, torso)

    # 3.5 Add visual meshes from URDF.
    visual_meshes = collect_visual_meshes(urdf)

    # Extra decorative meshes that are separate links in the URDF but were
    # attached as geoms in the original MuJoCo model (head, logo, hands, etc.)
    EXTRA_VISUALS = {
        'torso_link': [
            ('head_link', '0.0039635 0 -0.054', '1 0 0 0', '0.2 0.2 0.2 1'),
            ('logo_link', '0.0039635 0 -0.054', '1 0 0 0', '0.2 0.2 0.2 1'),
            ('waist_support_link', '0.0039635 0 -0.054', '1 0 0 0', '0.7 0.7 0.7 1'),
        ],
        'pelvis': [
            ('pelvis_contour_link', '0 0 0', None, '0.7 0.7 0.7 1'),
        ],
        'left_wrist_yaw_link': [
            ('left_rubber_hand', '0.0415 0.003 0', '1 0 0 0', '0.7 0.7 0.7 1'),
        ],
        'right_wrist_yaw_link': [
            ('right_rubber_hand', '0.0415 -0.003 0', '1 0 0 0', '0.7 0.7 0.7 1'),
        ],
    }

    if visual_meshes:
        # Insert <asset> before worldbody.
        asset = root.find('asset')
        if asset is None:
            asset = ET.Element('asset')
            idx = list(root).index(worldbody)
            root.insert(idx, asset)

        # Gather unique mesh files and add mesh assets.
        mesh_files = {}
        for link_name, items in visual_meshes.items():
            for mesh_name, pos, quat, fname in items:
                mesh_files[mesh_name] = fname
        for body_name, items in EXTRA_VISUALS.items():
            for mesh_name, pos, quat, rgba in items:
                mesh_files[mesh_name] = mesh_name + '.STL'
        for mesh_name, fname in sorted(mesh_files.items()):
            m = ET.SubElement(asset, 'mesh')
            m.set('name', mesh_name)
            m.set('file', fname)

        # Add a visual-only mesh geom to each body that has a URDF visual.
        # Hide the original primitive collision geoms from the default render group.
        bodies_with_visuals = set(visual_meshes.keys()) | set(EXTRA_VISUALS.keys())
        for body in root.iter('body'):
            name = body.get('name')
            added = False
            if name in visual_meshes:
                for mesh_name, pos, quat, _ in visual_meshes[name]:
                    geom = ET.SubElement(body, 'geom')
                    geom.set('type', 'mesh')
                    geom.set('mesh', mesh_name)
                    geom.set('pos', pos)
                    if quat is not None:
                        geom.set('quat', quat)
                    geom.set('contype', '0')
                    geom.set('conaffinity', '0')
                    geom.set('rgba', '0.85 0.85 0.85 1')
                added = True
            if name in EXTRA_VISUALS:
                for mesh_name, pos, quat, rgba in EXTRA_VISUALS[name]:
                    geom = ET.SubElement(body, 'geom')
                    geom.set('type', 'mesh')
                    geom.set('mesh', mesh_name)
                    geom.set('pos', pos)
                    if quat is not None:
                        geom.set('quat', quat)
                    geom.set('contype', '0')
                    geom.set('conaffinity', '0')
                    geom.set('rgba', rgba)
                added = True
            # Make existing primitive geoms invisible in default rendering.
            if added:
                for geom in body.findall('geom'):
                    # Only affect primitive geoms (mesh visual geoms have no 'type' default issue)
                    if geom.get('type') != 'mesh':
                        geom.set('group', '3')

    # 4. Extension and sensors
    extension = root.find('extension')
    if extension is None:
        extension = ET.Element('extension')
        root.insert(0, extension)
    plugin_ext = ET.SubElement(extension, 'plugin')
    plugin_ext.set('plugin', 'mujoco.sensor.ray_caster_camera')

    sensor = root.find('sensor')
    if sensor is None:
        sensor = ET.Element('sensor')
        root.append(sensor)

    fq = ET.SubElement(sensor, 'framequat')
    fq.set('name', 'imu_quat')
    fq.set('objtype', 'site')
    fq.set('objname', 'imu')

    gyro = ET.SubElement(sensor, 'gyro')
    gyro.set('name', 'imu_gyro')
    gyro.set('site', 'imu')

    acc = ET.SubElement(sensor, 'accelerometer')
    acc.set('name', 'imu_acc')
    acc.set('site', 'imu')

    plugin_sensor = ET.SubElement(sensor, 'plugin')
    plugin_sensor.set('name', 'depth_camera_sensor')
    plugin_sensor.set('plugin', 'mujoco.sensor.ray_caster_camera')
    plugin_sensor.set('objtype', 'camera')
    plugin_sensor.set('objname', 'depth_cam')
    configs = [
        ('focal_length', '1'),
        ('horizontal_aperture', '1.98297'),
        ('vertical_aperture', '1.11524'),
        ('size', '64 36'),
        ('dis_range', '0.30 3.0'),
        ('n_step_update', '17'),
        ('sensor_data_types', 'distance_to_image_plane'),
    ]
    for key, value in configs:
        cfg = ET.SubElement(plugin_sensor, 'config')
        cfg.set('key', key)
        cfg.set('value', value)

    # 5. Actuators
    actuator = root.find('actuator')
    if actuator is None:
        actuator = ET.Element('actuator')
        root.append(actuator)

    effort_limits = {
        'hip_pitch': 88, 'hip_roll': 139, 'hip_yaw': 88, 'knee': 139,
        'ankle_pitch': 50, 'ankle_roll': 50,
        'waist_yaw': 88, 'waist_roll': 50, 'waist_pitch': 50,
        'shoulder_pitch': 25, 'shoulder_roll': 25, 'shoulder_yaw': 25,
        'elbow': 25,
        'wrist_roll': 25, 'wrist_pitch': 5, 'wrist_yaw': 5,
    }

    for body in root.iter('body'):
        for joint in body.findall('joint'):
            name = joint.get('name')
            if name == 'floating_base_joint':
                continue
            key = None
            for k in effort_limits:
                if k in name:
                    key = k
                    break
            if key is None:
                continue
            limit = effort_limits[key]
            motor = ET.SubElement(actuator, 'motor')
            motor.set('name', name)
            motor.set('joint', name)
            motor.set('ctrlrange', f'-{limit} {limit}')

    # 6. Floor is provided by scene_29dof.xml; do NOT add one here.

    ET.indent(tree, space='  ')
    tree.write(dst, encoding='utf-8', xml_declaration=True)
    print(f'Wrote adapted MJCF to {dst}')


if __name__ == '__main__':
    main()
