import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Simplified AMV simulation from synthetic geostationary imagery
# Inspired by MTSAT/JMA-style cloud tracking
# ============================================================

# -----------------------------
# 1. Synthetic cloud field
# -----------------------------
def gaussian_blob(nx, ny, cx, cy, sx, sy, amp):
    y, x = np.mgrid[0:ny, 0:nx]
    return amp * np.exp(-(((x - cx) ** 2) / (2 * sx ** 2) +
                          ((y - cy) ** 2) / (2 * sy ** 2)))

def make_cloud_scene(nx=240, ny=180, seed=42):
    rng = np.random.default_rng(seed)
    img = np.zeros((ny, nx), dtype=float)

    # Add random cloud blobs
    for _ in range(18):
        cx = rng.uniform(0, nx)
        cy = rng.uniform(0, ny)
        sx = rng.uniform(6, 20)
        sy = rng.uniform(6, 20)
        amp = rng.uniform(0.4, 1.0)
        img += gaussian_blob(nx, ny, cx, cy, sx, sy, amp)

    # Add broad-scale structure
    y, x = np.mgrid[0:ny, 0:nx]
    img += 0.15 * np.sin(2 * np.pi * x / nx * 2.5)
    img += 0.10 * np.cos(2 * np.pi * y / ny * 1.5)

    # Normalize
    img -= img.min()
    img /= img.max() + 1e-12
    return img

# -----------------------------
# 2. Wind field for advection
# -----------------------------
def wind_field(nx, ny):
    y, x = np.mgrid[0:ny, 0:nx]

    # Background easterly flow + weak vortex-like perturbation
    u = 2.5 + 1.2 * np.sin(2 * np.pi * y / ny)
    v = 0.8 * np.cos(2 * np.pi * x / nx)

    # Add a gentle cyclonic circulation
    cx, cy = 0.65 * nx, 0.45 * ny
    dx = x - cx
    dy = y - cy
    r2 = dx**2 + dy**2 + 200.0
    u += -35.0 * dy / r2
    v +=  35.0 * dx / r2

    return u, v

# -----------------------------
# 3. Semi-Lagrangian advection
# -----------------------------
def bilinear_sample(img, x, y):
    ny, nx = img.shape

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, nx - 1)
    x1 = np.clip(x1, 0, nx - 1)
    y0 = np.clip(y0, 0, ny - 1)
    y1 = np.clip(y1, 0, ny - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    return (wa * img[y0, x0] +
            wb * img[y0, x1] +
            wc * img[y1, x0] +
            wd * img[y1, x1])

def advect(img, u, v, dt=1.0):
    ny, nx = img.shape
    y, x = np.mgrid[0:ny, 0:nx]

    # Backward trajectories
    x_src = x - u * dt
    y_src = y - v * dt

    # Clip to domain
    x_src = np.clip(x_src, 0, nx - 1)
    y_src = np.clip(y_src, 0, ny - 1)

    out = bilinear_sample(img, x_src, y_src)
    return out

# -----------------------------
# 4. Simple AMV retrieval by block matching
# -----------------------------
def normalized_cc(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a*a) * np.sum(b*b)) + 1e-12
    return np.sum(a * b) / denom

def estimate_amvs(frame1, frame2, win=15, search=6, stride=16):
    ny, nx = frame1.shape
    half = win // 2

    xs = []
    ys = []
    us = []
    vs = []
    qis = []

    for cy in range(half + search, ny - half - search, stride):
        for cx in range(half + search, nx - half - search, stride):
            template = frame1[cy-half:cy+half+1, cx-half:cx+half+1]

            best_cc = -1.0
            best_dx = 0
            best_dy = 0

            for dy in range(-search, search + 1):
                for dx in range(-search, search + 1):
                    patch = frame2[cy+dy-half:cy+dy+half+1,
                                   cx+dx-half:cx+dx+half+1]
                    cc = normalized_cc(template, patch)
                    if cc > best_cc:
                        best_cc = cc
                        best_dx = dx
                        best_dy = dy

            # crude quality control
            if best_cc > 0.55:
                xs.append(cx)
                ys.append(cy)
                us.append(best_dx)   # motion in x
                vs.append(best_dy)   # motion in y
                qis.append(best_cc)  # use correlation as proxy QI

    return np.array(xs), np.array(ys), np.array(us), np.array(vs), np.array(qis)

# -----------------------------
# 5. Main run
# -----------------------------
def main():
    nx, ny = 240, 180

    # Initial synthetic satellite image
    img0 = make_cloud_scene(nx, ny, seed=7)

    # Wind field (pixels per timestep)
    u, v = wind_field(nx, ny)

    # Three successive images, echoing the 3-image AMV concept
    img1 = advect(img0, u, v, dt=1.0)
    img2 = advect(img1, u, v, dt=1.0)

    # Add slight radiance-like noise
    rng = np.random.default_rng(123)
    img0n = np.clip(img0 + 0.02 * rng.standard_normal(img0.shape), 0, 1)
    img1n = np.clip(img1 + 0.02 * rng.standard_normal(img1.shape), 0, 1)
    img2n = np.clip(img2 + 0.02 * rng.standard_normal(img2.shape), 0, 1)

    # Retrieve AMVs between successive frames
    xs, ys, us, vs, qis = estimate_amvs(img0n, img1n, win=15, search=6, stride=16)

    # Compare with "truth"
    u_true = u[ys.astype(int), xs.astype(int)]
    v_true = v[ys.astype(int), xs.astype(int)]
    speed_err = np.sqrt((us - u_true)**2 + (vs - v_true)**2)

    print("Retrieved vectors:", len(xs))
    if len(xs) > 0:
        print("Mean speed error (pixels/timestep): {:.3f}".format(np.mean(speed_err)))
        print("Median speed error (pixels/timestep): {:.3f}".format(np.median(speed_err)))
        print("Mean quality indicator proxy: {:.3f}".format(np.mean(qis)))

    # -----------------------------
    # Plot imagery + AMVs
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    axes[0].imshow(img0n, cmap="gray", origin="lower")
    axes[0].set_title("Synthetic MTSAT-like Image t0")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].imshow(img1n, cmap="gray", origin="lower")
    axes[1].set_title("Synthetic MTSAT-like Image t1")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    axes[2].imshow(img1n, cmap="gray", origin="lower")
    axes[2].quiver(xs, ys, us, vs, qis, scale=35, pivot='mid')
    axes[2].set_title("Retrieved AMVs on t1")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    plt.show()

    # -----------------------------
    # Plot error histogram
    # -----------------------------
    if len(speed_err) > 0:
        plt.figure(figsize=(7, 4))
        plt.hist(speed_err, bins=20, edgecolor='black')
        plt.title("AMV Speed Error Histogram")
        plt.xlabel("Vector error (pixels/timestep)")
        plt.ylabel("Count")
        plt.show()

if __name__ == "__main__":
    main()
