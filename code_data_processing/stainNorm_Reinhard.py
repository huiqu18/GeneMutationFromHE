
import numpy as np
import skimage.color as color


def lab_split(I):
    """
    Convert from RGB uint8 to LAB and split into channels
    """
    I = color.rgb2lab(I)
    I = I.astype(np.float32)
    I1, I2, I3 = I[:, :, 0], I[:, :, 1], I[:, :, 2]
    return I1, I2, I3


def merge_back(I1, I2, I3):
    """
    Take seperate LAB channels and merge back to give RGB uint8
    """
    I1 = np.clip(I1, 0, 100)
    I2 = np.clip(I2, -128, 127)
    I3 = np.clip(I3, -128, 127)
    I = np.stack((I1, I2, I3), axis=2).astype(np.float64)
    I = color.lab2rgb(I) * 255
    return I.astype(np.uint8)


def get_mean_std(I, mask=None):
    """
    Get mean and standard deviation of each channel
    """
    I1, I2, I3 = lab_split(I)
    if mask is not None:
        I1, I2, I3 = I1[mask > 0], I2[mask > 0], I3[mask > 0]
    means = np.mean(I1), np.mean(I2), np.mean(I3)
    stds = np.std(I1), np.std(I2), np.std(I3)
    return means, stds


def standardize_brightness(I):
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


class normalizer(object):
    """
    A stain normalization object
    """

    def __init__(self, means, stds):
        self.target_means = means
        self.target_stds = stds

    def transform(self, I, means, stds):
        I1, I2, I3 = lab_split(I)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return merge_back(norm1, norm2, norm3)