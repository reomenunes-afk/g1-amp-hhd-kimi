**Languages:** 
[English](compute.md) | [简体中文](compute.zh-CN.md)
# Computation Model

## 1. Loss Angle
![](./image/loss_angle.svg)

Let $\theta$ be the angle between the ray emitted from the Raycaster or LDM (light detetion and ranging moudle) and the surface normal at the hit position.

*   **Condition**: When $\theta > \text{lossangle}$, data loss occurs.

---

## 2. Stereo Model
![](./image/stereo_camera_ray_model.svg)
![](./image/stereo_camera_ray_model_loss.svg)
![](./image/baseline_real.png)

This model simulates **data shadows** typically found in real stereo camera ranging:

1.  **Ray Emission**: A ray emitted from the Raycaster/LDM hits an object surface, establishing a "Hit Point".
2.  **Back-tracing**: Rays are mathematically cast from both the `left_camera` and `right_camera` towards this Hit Point.
3.  **Distance Verification**:
    *   If the measured distance of these rays matches the Euclidean distance from the camera to the Hit Point, the data is considered **valid**.
    *   If the measured data for any of the rays does not match (indicating occlusion), **data loss** occurs.

---

## 3. Stereo Noise & Min Energy
![](./image/stereo_camera_ray_model_noise.svg)

This noise model is derived from energy analysis:

*   **Physical Principle**: When a ray from the Raycaster/LDM hits a surface, the intensity of the reflected light decreases as the angle between the line of sight and the surface normal increases. Due to specular reflection characteristics, much of the energy is scattered away, reducing the intensity received by the Stereo Camera.
*   **Observation**: Depth data is most stable when viewing a surface perpendicularly. When viewing at an oblique angle, significant noise is generated, increasing the likelihood of data loss.

A probabilistic model is established to simulate this data loss:

### Calculation Logic

1.  **Calculate Cosine Similarity**:
    $\theta_1$ and $\theta_2$ represent the angles between the return path vectors (to the Stereo Camera) and the surface normal at the hit point. These are obtained by calculating the cosine similarity $\cos(\theta)$ between the `stereo_ray_normal` and the `hit_face_normal`.

2.  **Calculate Energy ($E$)**:     
    $$E = \min(\cos(\theta_1), \cos(\theta_2))$$

3.  **Threshold Check & Probability Calculation**:
    The calculated energy $E$ is compared with the parameter `min_energy`:      
    *   **Case 1**: If E <= min_energy, the data is lost immediately.       
    *   **Case 2**: If E > min_energy, the probability of loss ($P_{loss}$) is calculated.      

    The noise model parameter `pow` is the exponent applied to the normalized inverted energy:      

    $$X = 1 - \frac{E}{1 - min_energy}$$

    $$P_{loss} = X^{pow}$$

![](./image/Ploss.svg)


## 4. Domain Randomization
*   **Domain Randomization Scheme**: Based on the model described above, the parameters `min_energy` and `pow` can be randomized.
*   **Further Extension**: Parameters can be allocated based on `geom_id`, allowing for specific assignment of `(min_energy_range, pow)`.

## 5. IsaacLab stereo_vision_ray_caster_camera
*   **Status**: Pending Testing

## 6. mjlab stereo_vision_ray_caster_camera
*   **Status**: Pending Testing