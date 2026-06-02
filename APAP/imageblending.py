import numpy as np
import cv2 as cv


def imageblending(warped_img1, warped_img2, w1, w2):

    mass = w1 + w2
    mass[mass == 0] = np.nan

    # If a pixel value is non-zero then its weight should be max, otherwise, the other should be maximised.

    output_canvas = np.zeros_like(warped_img1)

    for c in range(warped_img1.shape[2]):
       output_canvas[:, :, c] = ((warped_img1[:, :, c] * w1) + (warped_img2[:, :, c] * w2)) / mass

    return output_canvas
