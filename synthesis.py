import numpy as np
import cv2
import random

t_exposure = 1/2400
t_vlcs = [1/1202]
snrs = [-10, -11, -12, -13, -14]
e_c = 1.32
n_channel = 3

fx = 585.0  # Focal length in pixels (horizontal)
fy = 585.0  # Focal length in pixels (vertical)
cx = 324.5  # Principal point (horizontal)
cy = 253.7  # Principal point (vertical)


def script_fsk_depth_k0kp(image, alb, depth, stripes, mode, shape, seed, normal=None, Onorm_module=True):
    state_random = random.getstate()
    state_numpy = np.random.get_state()

    if mode != 'train':
        random.seed(seed)
        np.random.seed(seed)

    gamma = 0.45
    mindepth = 0.5

    depth[np.isnan(depth)] = mindepth
    depth[np.isinf(depth)] = 100
    mask = (depth <= mindepth).astype(np.uint8)
    depth = cv2.inpaint(depth.astype(np.float32), mask, inpaintRadius=8, flags=cv2.INPAINT_TELEA)
    depth[depth<=mindepth] = mindepth

    # choose a stripe pattern
    stripe = stripes[np.random.randint(0, stripes.shape[0]), :]

    albo = alb
    albo[albo == 0] = 1
    shading = image.astype(np.float64) / albo.astype(np.float64)
    shading = np.power(shading, 1/gamma)

    # stretch stripe in channel direction
    stripe = stripe[:, np.newaxis]
    t1 = np.tile(stripe, (1, shape[1]))

    t1 = np.tile(t1[:, :, np.newaxis], (1, 1, 3))

    # suppose the light place
    light_position = np.array([0.0, 0.0, 0.0])

    height, width = image.shape[:2]
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    Z = depth
    X = (x_coords - cx) * Z / fx
    Y = (y_coords - cy) * Z / fy
    world_coords = np.dstack((X, Y, Z))  # (height, width, 3)

    # compute light vector
    light_vectors = world_coords - light_position
    light_distances = np.linalg.norm(light_vectors, axis=2, keepdims=True)
    light_vectors_normalized = light_vectors / light_distances  # 归一化光线向量

    final_intensity = 1.0

    if not Onorm_module:
        normal_map = normal

        # get pixel's normal vector
        normal_vectors = normal_map

        # compute diffuse factor
        final_intensity = np.maximum(0, np.sum(normal_vectors * light_vectors_normalized, axis=2, keepdims=True))

    temp = final_intensity * t1 / (light_distances ** 2)

    k0 = 1200
    kp = 1

    O_norm = (np.power(np.mean(temp * k0, axis=2), gamma)[:,:,np.newaxis] * alb).astype(np.uint8)
    O_norm = np.clip(O_norm, 0, 255)

    shading_final = shading + temp * k0 * kp

    striped_frame = alb.astype(np.float64) * np.power(shading_final, gamma)

    striped_frame = np.clip(striped_frame, 0, 255)
    striped_frame = striped_frame.astype(np.uint8)

    random.setstate(state_random)
    np.random.set_state(state_numpy)

    return image, striped_frame, O_norm


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

