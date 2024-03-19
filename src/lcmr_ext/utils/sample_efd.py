import cv2
import torch
import numpy as np
from imageio.v3 import imread
from pyefd import elliptic_fourier_descriptors, reconstruct_contour, normalize_efd


order = 32


heart_image = imread("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Heart_coraz%C3%B3n.svg/1200px-Heart_coraz%C3%B3n.svg.png")[..., 0]
heart_contour = cv2.findContours((heart_image).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0][:, 0, :]
heart_efd = elliptic_fourier_descriptors(heart_contour, order=order, normalize=True)
heart_efd[0, 1:3] = 0
heart_contour = reconstruct_contour(heart_efd)


square_contour = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
square_efd = elliptic_fourier_descriptors(square_contour, order=order, normalize=True)
square_efd[0, 1:3] = 0
square_contour = reconstruct_contour(square_efd)


ellipse_efd = np.zeros_like(square_efd)
ellipse_efd[0, 0] = 1.0
ellipse_efd[0, 3] = -0.5
ellipse_efd[0, 1:3] = 0
ellipse_efd = normalize_efd(ellipse_efd)
ellipse_contour = reconstruct_contour(ellipse_efd)


heart_efd = torch.from_numpy(heart_efd).to(torch.float32)
square_efd = torch.from_numpy(square_efd).to(torch.float32)
ellipse_efd = torch.from_numpy(ellipse_efd).to(torch.float32)
