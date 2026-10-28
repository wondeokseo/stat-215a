import numpy as np

Y_COL = 0
X_COL = 1


def shift_images_to_zero(images):
    """Shift images to have minimums of X and Y coordinates be 0

    Args:
        images:
            Images from MISR .npz files

    Returns:
        Shifted images
    """

    shifted_images = []
    for image in images:
        image = image.copy()
        image[:, Y_COL] -= image[:, Y_COL].min()
        image[:, X_COL] -= image[:, X_COL].min()

        shifted_images.append(image)

    return shifted_images

def crop_images(images):
    """Crop images to standardized shape with all valid pixels

    Args:
        images:
            Images from MISR .npz files

    Returns:
        Cropped images
    """

    max_x = max(int(image[:, X_COL].max()) for image in images)
    max_y = max(int(image[:, Y_COL].max()) for image in images)

    heatmap = np.zeros((max_x + 1, max_y + 1)) 

    for image in images:
        x = image[:, X_COL].astype(np.int64)
        y = image[:, Y_COL].astype(np.int64)

        mask = np.zeros_like(heatmap, dtype=bool)
        mask[x, y] = True
        heatmap += mask

    coords_in_all = np.array(np.where(heatmap == len(images)))

    top_side = coords_in_all[:, coords_in_all[1] == coords_in_all[1].max()]
    bottom_side = coords_in_all[:, coords_in_all[1] == coords_in_all[1].min()]

    left_x = max(top_side[0].min(), bottom_side[0].min())
    right_x = min(top_side[0].max(), bottom_side[0].max())

    cropped_images = []
    for image in images:
        valid_pixels_mask = (image[:, X_COL] >= left_x) & (image[:, X_COL] <= right_x)
        cropped_image = image[valid_pixels_mask]
        cropped_image[:, X_COL] -= left_x

        cropped_images.append(cropped_image)

    return cropped_images


def clean_data(images, crop_all_images=False):
    """Returns cleaned images

    Args:
        images:
            Raw images from MISR .npz files
        crop_all_images:
            Judgment call - Crop images to standardized shape (297x382)
            with all valid pixels

    Returns:
        Cleaned images
    """

    images = shift_images_to_zero(images)

    if crop_all_images:
        images = crop_images(images)

    return images
