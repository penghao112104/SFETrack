import cv2


def _read_rgb(path: str, name: str):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read {name} image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _resize_to_rgb(rgb, image, image_path=None):
    if image.shape[:2] == rgb.shape[:2]:
        return image
    old_shape = image.shape[:2]
    image = cv2.resize(image, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    if image_path is not None:
        print(f"[WARN] resize TIR to RGB size: {image_path}, {old_shape} -> {rgb.shape[:2]}")
    return image


def get_x_frame(color_path, x_path, dtype='rgbrgb', depth_clip=False):
    """Read one RGB frame and one TIR frame.

    The project uses dtype='rgbrgb': RGB and TIR are both converted to RGB-like
    three-channel images and concatenated into a six-channel array.
    """
    if dtype != 'rgbrgb':
        raise ValueError(f"Unsupported dtype for this RGBT project: {dtype}")

    rgb = _read_rgb(color_path, "RGB")
    tir = _read_rgb(x_path, "TIR")
    tir = _resize_to_rgb(rgb, tir, x_path)
    return cv2.merge((rgb, tir))
