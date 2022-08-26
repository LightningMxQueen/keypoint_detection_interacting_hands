import numpy as np

def world2cam(world_coord, R, T):
    """Transform world coordinates to cam coordinates"""
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord

def cam2pixel(cam_coord, f, c):
    """Project 3D coords onto 2D image"""
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord


