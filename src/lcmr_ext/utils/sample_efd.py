import cv2
import numpy as np
import torch
from imageio.v3 import imread
from pyefd import elliptic_fourier_descriptors, normalize_efd, reconstruct_contour

# Heart

heart_image = imread("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Heart_coraz%C3%B3n.svg/1200px-Heart_coraz%C3%B3n.svg.png")[..., 0]
heart_contour = cv2.findContours((heart_image).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0][:, 0, :]


def heart_efd(order: int = 64):
    heart_efd = elliptic_fourier_descriptors(heart_contour, order=order, normalize=True)
    heart_efd[0, 1:3] = 0
    return torch.from_numpy(heart_efd).to(torch.float32)


heart_contour = reconstruct_contour(heart_efd().numpy())

# Square

square_contour = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)


def square_efd(order: int = 64):
    square_efd = elliptic_fourier_descriptors(square_contour, order=order, normalize=True)
    square_efd[0, 1:3] = 0
    return torch.from_numpy(square_efd).to(torch.float32)


square_contour = reconstruct_contour(square_efd().numpy())

# Ellipse


def ellipse_efd(order: int = 64):
    ellipse_efd = np.zeros((order, 4))
    ellipse_efd[0, 0] = 1.0
    ellipse_efd[0, 3] = -0.5
    ellipse_efd[0, 1:3] = 0
    ellipse_efd = normalize_efd(ellipse_efd)
    return torch.from_numpy(ellipse_efd).to(torch.float32)


ellipse_contour = reconstruct_contour(ellipse_efd().numpy())

# Hourglass

h = np.sqrt(3) / 2


def hourglass_efd(order: int = 64):
    hourglass_efd = elliptic_fourier_descriptors([[0, 0], [0.5, h], [0, 2 * h], [1, 2 * h], [0.5, h], [1, 0], [0, 0]], order=order, normalize=True)
    hourglass_efd[0, 1:3] = 0
    return torch.from_numpy(hourglass_efd).to(torch.float32)


hourglass_contour = reconstruct_contour(hourglass_efd().numpy())

# Triangle


def triangle_efd(order: int = 64):
    triangle_efd = elliptic_fourier_descriptors([[0, 0], [0.5, h], [1, 0], [0, 0]], order=order)
    triangle_efd[0, 1:3] = 0
    return torch.from_numpy(triangle_efd).to(torch.float32)


triangle_contour = reconstruct_contour(triangle_efd().numpy())

# L


def L_efd(order: int = 64):
    L_efd = elliptic_fourier_descriptors([[0, 0], [0, 2], [1, 2], [1, 1], [3, 1], [3, 0], [0, 0]], order=order, normalize=True)
    L_efd[0, 1:3] = 0
    return torch.from_numpy(L_efd).to(torch.float32)


L_contour = reconstruct_contour(L_efd().numpy())
