import numpy as np

from utils.util_calculate_psnr_ssim import calculate_psnrb


def test_a_width_past_a_multiple_of_eight_keeps_the_last_block_edge():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(16, 17, 3), dtype=np.uint8)
    other = image.copy()
    other[:, 8, :] = 0
    score = calculate_psnrb(image, other, crop_border=0)
    # Edges sit after columns 7 and 15. Counting only (17 // 8 - 1) scores 13.15.
    assert abs(score - 18.0116) < 1e-3


def test_a_side_of_nine_is_not_minus_infinity():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(9, 9, 3), dtype=np.uint8)
    other = image.copy()
    other[:, 4, :] = 0
    score = calculate_psnrb(image, other, crop_border=0)
    # (9 // 8 - 1) is 0, so the old average divided by zero and returned -inf.
    assert abs(score - 10.0985) < 1e-3
