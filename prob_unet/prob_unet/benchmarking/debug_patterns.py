import numpy as np
from scipy.ndimage import gaussian_filter


def _norm01(img):
    img = img.astype(np.float32)
    lo, hi = np.min(img), np.max(img)
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img)
    return img

def _add(img, mask, value, mode="max"):
    if mode == "max":
        img[mask] = np.maximum(img[mask], value)
    elif mode == "add":
        img[mask] += value
    elif mode == "set":
        img[mask] = value
    return img

def _grid(h, w):
    y, x = np.mgrid[0:h, 0:w]
    return y, x

def add_circle(img, y0, x0, r, value, mode="max"):
    h, w = img.shape
    y, x = _grid(h, w)
    mask = (y - y0)**2 + (x - x0)**2 <= r**2
    return _add(img, mask, value, mode), {"type": "circle", "center": (y0, x0), "r": r, "value": value}

def add_ellipse(img, y0, x0, ry, rx, value, angle_deg=0, mode="max"):
    h, w = img.shape
    y, x = _grid(h, w)
    yc = y - y0
    xc = x - x0
    theta = np.deg2rad(angle_deg)
    ct, st = np.cos(theta), np.sin(theta)
    yr =  ct*yc + st*xc
    xr = -st*yc + ct*xc
    mask = (yr/ry)**2 + (xr/rx)**2 <= 1.0
    return _add(img, mask, value, mode), {"type": "ellipse", "center": (y0, x0), "ry": ry, "rx": rx, "angle": angle_deg, "value": value}

def add_rectangle(img, y0, x0, height, width, value, angle_deg=0, mode="max"):
    h, w = img.shape
    y, x = _grid(h, w)
    yc = y - y0
    xc = x - x0
    theta = np.deg2rad(angle_deg)
    ct, st = np.cos(theta), np.sin(theta)
    yr =  ct*yc + st*xc
    xr = -st*yc + ct*xc
    mask = (np.abs(yr) <= height/2) & (np.abs(xr) <= width/2)
    return _add(img, mask, value, mode), {"type": "rectangle", "center": (y0, x0), "h": height, "w": width, "angle": angle_deg, "value": value}

def add_line(img, y0, x0, y1, x1, thickness, value, mode="max"):
    """Anti-aliased-ish thick line by distance-to-segment threshold."""
    h, w = img.shape
    y, x = _grid(h, w)
    # vectorized distance from (x,y) to segment (x0,y0)-(x1,y1)
    vx, vy = x1 - x0, y1 - y0
    seg_len2 = vx*vx + vy*vy + 1e-9
    t = ((x - x0)*vx + (y - y0)*vy) / seg_len2
    t = np.clip(t, 0, 1)
    px = x0 + t*vx
    py = y0 + t*vy
    dist = np.hypot(x - px, y - py)
    mask = dist <= (thickness/2)
    return _add(img, mask, value, mode), {"type": "line", "p0": (y0, x0), "p1": (y1, x1), "th": thickness, "value": value}

def add_ring(img, y0, x0, r_inner, r_outer, value, mode="max"):
    h, w = img.shape
    y, x = _grid(h, w)
    d2 = (y - y0)**2 + (x - x0)**2
    mask = (d2 >= r_inner**2) & (d2 <= r_outer**2)
    return _add(img, mask, value, mode), {"type": "ring", "center": (y0, x0), "ri": r_inner, "ro": r_outer, "value": value}

def add_step_edge(img, orientation="vertical", pos=None, contrast=0.6):
    """Hard intensity step to create strong edges."""
    h, w = img.shape
    if pos is None:
        pos = (w // 2 if orientation == "vertical" else h // 2)
    if orientation == "vertical":
        img[:, pos:] = img[:, pos:] + contrast
        meta = {"type": "edge", "orientation": "vertical", "x": pos, "contrast": contrast}
    else:
        img[pos:, :] = img[pos:, :] + contrast
        meta = {"type": "edge", "orientation": "horizontal", "y": pos, "contrast": contrast}
    return img, meta

def add_gradient(img, axis="x", strength=0.5):
    h, w = img.shape
    if axis == "x":
        grad = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    elif axis == "y":
        grad = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    else:  # radial
        y, x = _grid(h, w)
        yc = (y - h/2) / (h/2)
        xc = (x - w/2) / (w/2)
        grad = np.sqrt(xc*xc + yc*yc)
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-9)
    img = img + strength * grad
    return img, {"type": "gradient", "axis": axis, "strength": strength}

def add_gabor(img, y0, x0, sigma, lambd, theta_deg, amplitude=0.5):
    """Gabor patch: localized oriented texture (edges + stripes)."""
    h, w = img.shape
    y, x = _grid(h, w)
    yc = y - y0
    xc = x - x0
    theta = np.deg2rad(theta_deg)
    ct, st = np.cos(theta), np.sin(theta)
    yr =  ct*yc + st*xc
    xr = -st*yc + ct*xc
    env = np.exp(-(xr**2 + yr**2) / (2*sigma**2))
    carrier = np.cos(2*np.pi*yr / (lambd + 1e-9))
    patch = amplitude * env * carrier
    img = img + patch
    return img, {"type": "gabor", "center": (y0, x0), "sigma": sigma, "lambda": lambd, "theta": theta_deg, "amp": amplitude}

def add_noise(img, gaussian_std=0.0, sp_prob=0.0, rng=None):
    rng = np.random.default_rng(rng)
    if gaussian_std > 0:
        img = img + rng.normal(0, gaussian_std, size=img.shape).astype(np.float32)
    if sp_prob > 0:
        mask = rng.random(img.shape) < sp_prob
        salt = rng.random(img.shape) < 0.5
        img = img.copy()
        img[mask & salt] = 1.0
        img[mask & ~salt] = 0.0
    return img

def generate_synthetic_image(
    shape=(256, 256),
    rng_seed=0,
    n_circles=(0, 3),
    n_ellipses=(0, 3),
    n_rects=(0, 3),
    n_lines=(0, 4),
    n_rings=(0, 2),
    n_gabors=(0, 2),
    add_bg_gradient=True,
    gaussian_blur_sigma=None,
    gaussian_noise_std=0.03,
    saltpepper_prob=0.0,
    normalize=True,
):
    """
    Returns: img (float32 HxW in [0,1] if normalize) and metadata (list of dicts).
    """
    rng = np.random.default_rng(rng_seed)
    h, w = shape
    img = np.zeros(shape, dtype=np.float32)
    meta = []

    if add_bg_gradient:
        axis = rng.choice(["x", "y", "radial"])
        strength = rng.uniform(0.1, 0.6)
        img, m = add_gradient(img, axis=axis, strength=strength)
        meta.append(m)

    def randint_or_range(lohi):
        if isinstance(lohi, tuple):
            return rng.integers(lohi[0], lohi[1] + 1)
        return int(lohi)

    # Circles
    for _ in range(randint_or_range(n_circles)):
        r = rng.integers(max(5, min(h, w)//20), max(8, min(h, w)//6))
        y0 = rng.integers(r, h - r)
        x0 = rng.integers(r, w - r)
        val = rng.uniform(0.3, 1.0)
        img, m = add_circle(img, y0, x0, r, val, mode=rng.choice(["max", "add", "set"]))
        meta.append(m)

    # Ellipses
    for _ in range(randint_or_range(n_ellipses)):
        ry = rng.integers(max(6, h//30), max(10, h//6))
        rx = rng.integers(max(6, w//30), max(10, w//6))
        y0 = rng.integers(ry, h - ry)
        x0 = rng.integers(rx, w - rx)
        ang = rng.integers(0, 180)
        val = rng.uniform(0.3, 1.0)
        img, m = add_ellipse(img, y0, x0, ry, rx, val, angle_deg=ang, mode=rng.choice(["max", "add", "set"]))
        meta.append(m)

    # Rectangles
    for _ in range(randint_or_range(n_rects)):
        hh = rng.integers(max(6, h//30), max(10, h//5))
        ww = rng.integers(max(6, w//30), max(10, w//5))
        y0 = rng.integers(hh//2, h - hh//2)
        x0 = rng.integers(ww//2, w - ww//2)
        ang = rng.integers(0, 180)
        val = rng.uniform(0.3, 1.0)
        img, m = add_rectangle(img, y0, x0, hh, ww, val, angle_deg=ang, mode=rng.choice(["max", "add", "set"]))
        meta.append(m)

    # Lines
    for _ in range(randint_or_range(n_lines)):
        y0, x0 = rng.integers(0, h), rng.integers(0, w)
        y1, x1 = rng.integers(0, h), rng.integers(0, w)
        thickness = rng.integers(1, max(2, min(h, w)//60))
        val = rng.uniform(0.4, 1.0)
        img, m = add_line(img, y0, x0, y1, x1, thickness, val, mode=rng.choice(["max", "add", "set"]))
        meta.append(m)

    # Rings
    for _ in range(randint_or_range(n_rings)):
        ro = rng.integers(max(8, min(h, w)//20), max(12, min(h, w)//5))
        ri = rng.integers(max(3, ro//4), ro - 2)
        y0 = rng.integers(ro, h - ro)
        x0 = rng.integers(ro, w - ro)
        val = rng.uniform(0.3, 1.0)
        img, m = add_ring(img, y0, x0, ri, ro, val, mode=rng.choice(["max", "add", "set"]))
        meta.append(m)

    # Gabors (localized oriented textures/edges)
    for _ in range(randint_or_range(n_gabors)):
        y0 = rng.integers(0, h)
        x0 = rng.integers(0, w)
        sigma = rng.uniform(min(h, w)*0.03, min(h, w)*0.12)
        lambd = rng.uniform(min(h, w)*0.04, min(h, w)*0.16)
        theta = rng.uniform(0, 180)
        amp = rng.uniform(0.2, 0.7)
        img, m = add_gabor(img, y0, x0, sigma, lambd, theta, amplitude=amp)
        meta.append(m)

    # A hard step edge somewhere (strong gradient)
    if rng.random() < 0.7:
        orientation = rng.choice(["vertical", "horizontal"])
        img, m = add_step_edge(img, orientation=orientation, contrast=rng.uniform(0.2, 0.8))
        meta.append(m)

    # Optional blur to vary edge sharpness
    if gaussian_blur_sigma is not None:
        img = gaussian_filter(img, gaussian_blur_sigma)
        meta.append({"type": "gaussian_blur", "sigma": gaussian_blur_sigma})

    # Noise
    img = add_noise(img, gaussian_noise_std, saltpepper_prob, rng=rng)

    if normalize:
        img = _norm01(img)

    return img.astype(np.float32), meta
