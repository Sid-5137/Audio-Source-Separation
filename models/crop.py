def centre_crop(x, target):
    """
    Center-crop 3D or 4D input tensor along the last two spatial dimensions so it fits the target tensor shape.
    :param x: Input tensor to be cropped.
    :param target: Tensor shape to match.
    :return: Cropped input tensor matching the spatial dimensions of the target.
    """
    if x is None:
        return None
    if target is None:
        return x

    diff_h = x.size(2) - target.size(2)
    diff_w = x.size(3) - target.size(3)

    # Adjust for odd differences by adding 1 pixel to the crop from one side
    crop_h1 = diff_h // 2
    crop_h2 = diff_h - crop_h1
    crop_w1 = diff_w // 2
    crop_w2 = diff_w - crop_w1

    # Crop the input tensor to match the target size
    return x[:, :, crop_h1:x.size(2) - crop_h2, crop_w1:x.size(3) - crop_w2].contiguous()
