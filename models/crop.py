# crop.py
def centre_crop(x, target):
    """
    Center-crop input tensor along the last two dimensions to match the target tensor shape.
    :param x: Input tensor
    :param target: Target tensor shape to match
    :return: Cropped input tensor
    """
    if x is None or target is None:
        return x

    crop_h = (x.size(-2) - target.size(-2)) // 2
    crop_w = (x.size(-1) - target.size(-1)) // 2

    if crop_h < 0 or crop_w < 0:
        raise ValueError("Cannot crop a tensor of smaller size to a larger size")

    return x[:, :, crop_h:x.size(-2) - crop_h, crop_w:x.size(-1) - crop_w]
