"""Integral-image helper to deliver O(1) region mean/variance queries."""

import numpy as np

class ImageAnalyzer:
    def __init__(self, image):
        # Precompute integral images (summed-area tables) for fast O(1) region stats.
        self.np_rgb = np.array(image.convert("RGB"), dtype=np.float32)
        ycbcr_image = image.convert('YCbCr')
        self.np_ycbcr = np.array(ycbcr_image, dtype=np.float32)
        # Prefix sums for Y, Cb, Cr channels and their squares (variance needs both sum and sum of squares).
        self.sum_ycbcr = np.pad(self.np_ycbcr.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0)), 'constant')
        self.sq_sum_ycbcr = np.pad((self.np_ycbcr ** 2).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0)), 'constant')
        self.sum_rgb = np.pad(self.np_rgb.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0)), 'constant')

    def get_region_stats(self, box):
        # Compute mean RGB and a lightweight variance proxy over a rectangular region using integral images.
        x1, y1, x2, y2 = box
        if x1 >= x2 or y1 >= y2: return (0, 0, 0), 0
        num_pixels = (x2 - x1) * (y2 - y1)
        # Mean RGB for block rendering.
        rgb_sum = self.sum_rgb[y2, x2] - self.sum_rgb[y1, x2] - self.sum_rgb[y2, x1] + self.sum_rgb[y1, x1]
        avg_color_rgb = tuple((rgb_sum / num_pixels).astype(int))
        # YCbCr variance guides splitting: brighter weight on luminance (Y * 2).
        ycbcr_sum = self.sum_ycbcr[y2, x2] - self.sum_ycbcr[y1, x2] - self.sum_ycbcr[y2, x1] + self.sum_ycbcr[y1, x1]
        ycbcr_sq_sum = self.sq_sum_ycbcr[y2, x2] - self.sq_sum_ycbcr[y1, x2] - self.sq_sum_ycbcr[y2, x1] + self.sq_sum_ycbcr[y1, x1]
        mean = ycbcr_sum / num_pixels
        mean_sq = ycbcr_sq_sum / num_pixels
        stds = np.sqrt(np.maximum(0, mean_sq - mean**2))
        variance = (stds[0] * 2) + stds[1] + stds[2]
        return avg_color_rgb, variance
