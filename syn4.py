import numpy as np
import cv2
import random
import time

from PIL import Image

# direction of stripe
# direction_horizontal = True

t_exposure = 1/2400
# t_vlcs = [1/1028, 1/1344, 1/1667, 1/1984]
t_vlcs = [1/1202]
snrs = [-10, -11, -12, -13, -14]
e_c = 1.32
n_channel = 3
# shape = [256, 256]

fx = 585.0  # Focal length in pixels (horizontal)
fy = 585.0  # Focal length in pixels (vertical)
cx = 324.5  # Principal point (horizontal)
cy = 253.7  # Principal point (vertical)

# 计算法线（通过深度图的梯度计算）
def compute_normals(depth_map, fx, fy, cx, cy):
    # Convert pixel coordinates to camera coordinates (X, Y, Z)
    height, width = depth_map.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    Z = depth_map
    X = (x_coords - cx) * Z / fx
    Y = (y_coords - cy) * Z / fy

    # Compute the gradients in world coordinates to estimate surface normals
    dx_world = (np.gradient(X, axis=1), np.gradient(X, axis=0))  # X direction gradient
    dy_world = (np.gradient(Y, axis=1), np.gradient(Y, axis=0))  # Y direction gradient

    # Calculate the cross product of gradients to get the normal vector
    normal_map = np.dstack((-dx_world[1], -dy_world[1], np.ones_like(depth_map)))

    # Normalize the normal map (make each normal vector unit length)
    norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_map /= norm  # Normalize each normal vector

    return normal_map

import numpy as np
import random

def script_fsk_depth(image, alb, depth, stripes, mode, shape, seed, direction_horizontal=True, normal=None, rough=None):
    state_random = random.getstate()
    state_numpy = np.random.get_state()

    if mode != 'train':
        random.seed(seed)
        np.random.seed(seed)

    gamma = 0.45

    mindepth = 1e-3
    depth[depth <= mindepth] = mindepth
    # mask = (depth < mindepth).astype(np.uint8)
    # depth = cv2.inpaint(depth.astype(np.float32), mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # print(image.shape)
    # print(depth.shape)
    # print(alb.shape)
    # print(normal.shape)
    # print(rough.shape)

    # 选择一个条纹
    stripe = stripes[np.random.randint(0, stripes.shape[0]), :]

    albo = alb
    albo[albo == 0] = 1
    shading = image.astype(np.float64) / albo.astype(np.float64)
    # shading = np.clip(shading, 0, 1)
    shading = np.power(shading, 1/gamma)

    # 按方向调整条纹
    if direction_horizontal:
        stripe = stripe[:, np.newaxis]
        t1 = np.tile(stripe, (1, shape[1]))  # 复制条纹，调整方向
    else:
        stripe = stripe[np.newaxis, :]
        t1 = np.tile(stripe, (shape[0], 1))

    t1 = np.tile(t1[:, :, np.newaxis], (1, 1, n_channel))

    # 光源位置 (假设为相机中心)
    light_position = np.array([0.0, 0.0, 0.0])

    # 创建像素坐标的网格
    height, width = image.shape[:2]
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # 将像素坐标转换为相机坐标 (X, Y, Z)
    Z = depth
    X = (x_coords - cx) * Z / fx
    Y = (y_coords - cy) * Z / fy
    world_coords = np.dstack((X, Y, Z))  # 形状: (height, width, 3)

    # 计算光线向量（从光源到每个像素点）
    light_vectors = world_coords - light_position
    light_distances = np.linalg.norm(light_vectors, axis=2, keepdims=True)
    light_vectors_normalized = light_vectors / light_distances  # 归一化光线向量

    final_intensity = 1.0

    if normal is not None:
        # 使用传入的 normal 和 rough
        normal_map = normal  # 假设 normal 已经是计算好的法线图
        rough_map = rough  # 假设 rough 是已经传入的粗糙度图

        # 获取每个像素的法线向量
        normal_vectors = normal_map

        # **漫反射光照计算**：
        diffuse_intensity = np.maximum(0, np.sum(normal_vectors * light_vectors_normalized, axis=2, keepdims=True))

        # **镜面反射计算**：
        view_vectors = light_vectors  # 假设相机在Z轴方向
        half_vector = light_vectors_normalized
        
        # 镜面反射强度计算，使用粗糙度（rough）调整反射的扩散程度
        # specular_intensity = np.maximum(0, np.sum(normal_vectors * half_vector, axis=2, keepdims=True)) ** (1.0 / (rough_map[:,:,np.newaxis] + 0.1))  # 使用粗糙度调整镜面反射
        specular_intensity = np.maximum(0, np.sum(normal_vectors * half_vector, axis=2, keepdims=True)) ** (1.0 / (rough_map[:,:,np.newaxis]/255.0 + 0.01)**2)

        # 结合漫反射和镜面反射
        final_intensity = diffuse_intensity + specular_intensity

    # **加入条纹样式**：
    temp = final_intensity * t1 / (light_distances ** 2)  # 将条纹样式应用到最终光照颜色上

    occ_layer = temp * alb.astype(np.float64)

    p_origin = np.sum(image.astype(np.float64))
    p_occ = np.sum(occ_layer)

    # Signal to noise ratio (SNR) adjustment
    snr = np.random.choice(snrs)
    ratio = 10 ** (snr / 10)
    weight = p_origin / p_occ * ratio

    shading_final = shading + temp * weight
    # shading_final = np.clip(shading_final, 0, 1)

    striped_frame = alb.astype(np.float64) * np.power(shading_final, gamma)

    # Apply the final striped light effect to the image
    # striped_frame = image.astype(np.float64) + occ_layer * weight

    striped_frame = np.clip(striped_frame, 0, 255)
    striped_frame = striped_frame.astype(np.uint8)

    random.setstate(state_random)
    np.random.set_state(state_numpy)

    return image, striped_frame, stripe


def script_fsk_depth_k0kp(image, alb, depth, stripes, mode, shape, seed, direction_horizontal=True, normal=None, rough=None, method=6):
    state_random = random.getstate()
    state_numpy = np.random.get_state()

    if mode != 'train':
        random.seed(seed)
        np.random.seed(seed)

    gamma = 0.45
    mindepth = 0.5

    # if np.isnan(depth).any():
    #     print(33)
    #     exit(0)

    # depth[depth <= mindepth] = mindepth
    # print(depth)
    depth[np.isnan(depth)] = mindepth
    depth[np.isinf(depth)] = 100
    mask = (depth <= mindepth).astype(np.uint8)
    depth = cv2.inpaint(depth.astype(np.float32), mask, inpaintRadius=8, flags=cv2.INPAINT_TELEA)
    depth[depth<=mindepth] = mindepth

    # 选择一个条纹
    stripe = stripes[np.random.randint(0, stripes.shape[0]), :]

    albo = alb
    albo[albo == 0] = 1
    shading = image.astype(np.float64) / albo.astype(np.float64)
    # shading = np.clip(shading, 0, 1)
    shading = np.power(shading, 1/gamma)

    # 按方向调整条纹
    if direction_horizontal:
        stripe = stripe[:, np.newaxis]
        t1 = np.tile(stripe, (1, shape[1]))  # 复制条纹，调整方向
    else:
        stripe = stripe[np.newaxis, :]
        t1 = np.tile(stripe, (shape[0], 1))

    t1 = np.tile(t1[:, :, np.newaxis], (1, 1, n_channel))

    # 光源位置 (假设为相机中心)
    light_position = np.array([0.0, 0.0, 0.0])

    # 创建像素坐标的网格
    height, width = image.shape[:2]
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # 将像素坐标转换为相机坐标 (X, Y, Z)
    Z = depth
    X = (x_coords - cx) * Z / fx
    Y = (y_coords - cy) * Z / fy
    world_coords = np.dstack((X, Y, Z))  # 形状: (height, width, 3)

    # 计算光线向量（从光源到每个像素点）
    light_vectors = world_coords - light_position
    light_distances = np.linalg.norm(light_vectors, axis=2, keepdims=True)
    light_vectors_normalized = light_vectors / light_distances  # 归一化光线向量

    final_intensity = 1.0

    if normal is not None and method == 6:
        # 使用传入的 normal 和 rough
        normal_map = normal  # 假设 normal 已经是计算好的法线图
        rough_map = rough  # 假设 rough 是已经传入的粗糙度图

        # 获取每个像素的法线向量
        normal_vectors = normal_map

        # **漫反射光照计算**：
        diffuse_intensity = np.maximum(0, np.sum(normal_vectors * light_vectors_normalized, axis=2, keepdims=True))

        # **镜面反射计算**：
        view_vectors = light_vectors  # 假设相机在Z轴方向
        half_vector = light_vectors_normalized
        
        # 镜面反射强度计算，使用粗糙度（rough）调整反射的扩散程度
        # specular_intensity = np.maximum(0, np.sum(normal_vectors * half_vector, axis=2, keepdims=True)) ** (1.0 / (rough_map[:,:,np.newaxis] + 0.1))  # 使用粗糙度调整镜面反射
        specular_intensity = np.maximum(0, np.sum(normal_vectors * half_vector, axis=2, keepdims=True)) ** (1.0 / (rough_map[:,:,np.newaxis]/255.0 + 0.01)**2)

        # 结合漫反射和镜面反射
        final_intensity = diffuse_intensity + specular_intensity

    # **加入条纹样式**：
    temp = final_intensity * t1 / (light_distances ** 2)  # 将条纹样式应用到最终光照颜色上

    if method == 9:
        k0 = 1200
    else:
        k0 = 600

    kp = 1

    if np.isnan(light_distances).any():
        print('6')
        exit(0)
    if np.isinf(temp).any():
        print('7')
        exit(0)

    O_norm = (np.power(np.mean(temp * k0, axis=2), gamma)[:,:,np.newaxis] * alb).astype(np.uint8)
    O_norm = np.clip(O_norm, 0, 255)

    shading_final = shading + temp * k0 * kp
    # shading_final = np.clip(shading_final, 0, 1)

    striped_frame = alb.astype(np.float64) * np.power(shading_final, gamma)

    # Apply the final striped light effect to the image
    # striped_frame = image.astype(np.float64) + occ_layer * weight

    striped_frame = np.clip(striped_frame, 0, 255)
    if np.isnan(striped_frame).any():
        print(8)
        exit(0)
    striped_frame = striped_frame.astype(np.uint8)

    random.setstate(state_random)
    np.random.set_state(state_numpy)

    return image, striped_frame, O_norm

def norm(data):
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    data_norm = data_norm.astype(np.uint8)

    return data_norm

def synthesize_fsk_stripe(t_exp=None, t_vlc=None, e_c=None, ncol=1080):
    if t_exp is None:
        # exposure time
        t_exp = 1 / 800
    if t_vlc is None:
        # cycle length of VLC emission
        t_vlc = 1 / ncol

    # stores the synthesized stripes
    stripes = np.zeros(ncol)

    # params
    t1 = (1 / (60 * e_c) - t_exp) / (ncol-1)

    # duty cycle is 50%
    t_light = t_vlc * 0.5
    
    # eliminate full cycles
    k = np.floor(t_exp / t_vlc)
    t_exp = t_exp - k * t_vlc

    t_start = t_exp * np.random.rand()

    for w in range(ncol):
        t_end = t_start + t_exp

        a_i = np.floor(t_start / t_vlc)
        b_i = np.floor(t_end / t_vlc)

        if a_i == b_i:
            b = a_i * t_vlc + t_light
            luminance = max(min(b - t_start, t_exp), 0)
        elif b_i == a_i + 1:
            b1 = a_i * t_vlc + t_light
            b2 = b_i * t_vlc
            luminance1 = max(b1 - t_start, 0)
            luminance2 = min(t_light, t_end - b2)
            luminance = luminance1 + luminance2
        else:
            luminance = 0  # case where b_i > a_i + 1

        stripes[w] = luminance
        t_start += t1

    stripes += k * t_light
    return stripes

def synthesize_multiple_stripe(stripe1, stripe2, place=0):
    if place == 0:
        place = random.randint(1, 1080)
    
    end = stripe1.shape[0]

    stripe1[place:end-1] = stripe2[place:end-1]

    return stripe1

if __name__ == '__main__':

    shape = [256, 256]
