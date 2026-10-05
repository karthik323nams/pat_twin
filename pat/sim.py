"""Virtual world: target motion, virtual camera viewport, disturbances."""
import numpy as np, cv2
from dataclasses import dataclass

@dataclass
class Cfg:
    screen: int = 2000
    res: tuple = (640, 480)
    fov: tuple = (4.0, 3.0)          # degrees
    fps: int = 30
    tsize: int = 10                  # target size px (5-20)
    motion: str = "circular"         # straight|circular|figure8|random
    speed: float = 250.0             # target speed px/s
    max_rate: float = 5.0            # max pan/tilt deg/s
    noise: tuple = ("sp", "gauss")   # sp, gauss, poisson
    sigma: float = 15.0              # gaussian std (<=20)
    sp_frac: float = 0.10
    jitter: float = 8.0              # +-px/frame
    platform: float = 5.0            # platform motion amplitude (px)
    weather: str = "clear"           # clear|haze|fog|rain|lowlight
    seed: int = 1
    duration: float = 30.0
    stars: int = 3000                # static star-field texture (gives stabilizer a reference)
    wide_bin: int = 8                # binning of the wide-area cue sensor
    wide_cue: bool = True            # wide-area acquisition cue on/off
    stabilize: bool = False          # image-domain jitter stabilizer
    use_imm: bool = True             # IMM multi-model tracker (CV, CA, CT)
    use_ai: bool = True              # AI CNN candidate verifier
    use_fsm: bool = True             # Fine Steering Mirror / fine pointing stage
    decoys: int = 0                  # Number of false decoy targets
    @property
    def ppd(self): return self.res[0] / self.fov[0]      # px per degree
    @property
    def dt(self): return 1.0 / self.fps

WEATHER = {"clear": (1, 0, 0), "haze": (.7, 15, 0), "fog": (.45, 30, 1),
           "rain": (.75, 5, 0), "lowlight": (.35, -8, 0)}

class Motion:
    def __init__(s, c, rng):
        s.c, s.rng, W = c, rng, c.screen
        s.R = 0.25 * W; s.w = c.speed / s.R; s.ph = rng.uniform(0, 2 * np.pi)
        s.p = rng.uniform(.2 * W, .8 * W, 2); a = rng.uniform(0, 2 * np.pi)
        s.v = c.speed * np.array([np.cos(a), np.sin(a)])
        # Initialize decoys if requested
        s.decoy_ph = [rng.uniform(0, 2 * np.pi) for _ in range(c.decoys)]
        s.decoy_pos = [rng.uniform(.25 * W, .75 * W, 2) for _ in range(c.decoys)]
        s.decoy_vel = [c.speed * 0.4 * np.array([np.cos(a), np.sin(a)]) for a in rng.uniform(0, 2 * np.pi, c.decoys)]

    def step(s, t):
        c, W = s.c, s.c.screen; m = c.motion
        if m == "circular":
            return W / 2 + s.R * np.array([np.cos(s.w * t + s.ph), np.sin(s.w * t + s.ph)])
        if m == "figure8":
            return W / 2 + np.array([s.R * np.sin(s.w * t + s.ph), s.R / 2 * np.sin(2 * (s.w * t + s.ph))])
        if m == "spiral":
            r_sp = s.R * (0.35 + 0.65 * (0.5 + 0.5 * np.sin(0.2 * s.w * t + s.ph)))
            return W / 2 + r_sp * np.array([np.cos(s.w * t + s.ph), np.sin(s.w * t + s.ph)])
        if m == "sinusoidal":
            return W / 2 + np.array([s.R * np.sin(s.w * t + s.ph), (s.R * 0.6) * np.sin(2.8 * s.w * t + s.ph)])
        if m == "leo":  # Keplerian orbit pass across sky (variable angular speed)
            tau = (t - c.duration / 2.0) / max(c.duration / 3.0, 1e-3)
            x = W / 2 + (0.38 * W) * np.tanh(tau)
            y = W / 2 - (0.32 * W) / np.cosh(tau) * np.cos(s.ph)
            return np.array([x, y])
        if m == "random":   # Ornstein-Uhlenbeck velocity
            s.v += -0.8 * (s.v - 0) * c.dt + c.speed * 1.5 * np.sqrt(c.dt) * s.rng.normal(size=2)
            n = np.linalg.norm(s.v); s.v *= min(1, 1.3 * c.speed / (n + 1e-9))
        s.p = s.p + s.v * c.dt                                # straight & random
        for i in range(2):                                    # bounce off screen edges
            m_edge = c.res[0] // 2 + 20
            if not m_edge < s.p[i] < W - m_edge: s.v[i] *= -1; s.p[i] = np.clip(s.p[i], m_edge + 1, W - m_edge - 1)
        return s.p.copy()

    def get_decoys(s, t):
        """Returns list of decoy positions (world coords)."""
        c, W = s.c, s.c.screen; res = []
        for i in range(c.decoys):
            p = s.decoy_pos[i] + s.decoy_vel[i] * t + 30.0 * np.sin(0.4 * t + s.decoy_ph[i])
            for k in range(2):
                p[k] = np.clip(p[k], 100, W - 100)
            res.append(p)
        return res

def apply_noise(img, c, rng):
    f = img.astype(np.float32)
    if "poisson" in c.noise: f = rng.poisson(np.clip(f, 0, None)).astype(np.float32)
    if "gauss" in c.noise: f += rng.normal(0, c.sigma, f.shape)
    f = np.clip(f, 0, 255).astype(np.uint8)
    if "sp" in c.noise:
        m = rng.random(f.shape); f[m < c.sp_frac / 2] = 0; f[m > 1 - c.sp_frac / 2] = 255
    return f

def weather(f, c, rng, scale=1):
    k, b, blur = WEATHER[c.weather]
    f = f * k + b
    if blur: f = cv2.GaussianBlur(f.astype(np.float32), (0, 0), 2.0 / scale)
    if c.weather == "rain" and scale == 1:
        for _ in range(40):
            x, y = rng.integers(0, f.shape[1]), rng.integers(0, f.shape[0])
            cv2.line(f, (x, y), (x + 3, y + 14), 90, 1)
    return f

def target_patch(img, x, y, c):
    h = c.tsize // 2
    cv2.rectangle(img, (int(x) - h, int(y) - h), (int(x) + h - 1, int(y) + h - 1), 230, -1)

def make_stars(c, rng):
    return rng.uniform(0, c.screen, (c.stars, 2)), rng.uniform(50, 120, c.stars).astype(np.float32)

def stars_layer(st, sv, tl, shape):
    h, w = shape; u = st - tl
    m = (u[:, 0] >= 0) & (u[:, 0] < w - 1) & (u[:, 1] >= 0) & (u[:, 1] < h - 1)
    L = np.zeros(shape, np.float32); ui = np.rint(u[m]).astype(int); L[ui[:, 1], ui[:, 0]] = sv[m]
    return cv2.GaussianBlur(L, (0, 0), 0.9) * 5.0

class Camera:
    """Renders what the virtual pan-tilt camera sees at pose `cam` (world px)."""
    def __init__(s, c, rng): s.c, s.rng = c, rng; s.st, s.sv = make_stars(c, np.random.default_rng(c.seed + 1000))
    def wide(s, targets):
        """Simulated wide-area cue sensor: whole screen, binned by wide_bin (noise std reduced by binning)."""
        c = s.c; b = c.wide_bin; n = c.screen // b; img = np.full((n, n), 25, np.float32)
        for p in targets:
            i = (np.asarray(p) / b).astype(int); img[i[1]:i[1] + 2, i[0]:i[0] + 2] = 25 + 205 * min(1, (c.tsize / b) ** 2 / 2)
        img = weather(img, c, s.rng, b); v2 = 0.0
        if "gauss" in c.noise: v2 += c.sigma ** 2
        if "sp" in c.noise: v2 += c.sp_frac / 2 * (230 ** 2 + 25 ** 2)
        if "poisson" in c.noise: v2 += 25
        return np.clip(img + s.rng.normal(0, np.sqrt(v2) / b, img.shape), 0, 255).astype(np.uint8)
    def offsets(s, t):
        c = s.c; j = s.rng.uniform(-c.jitter, c.jitter, 2)
        p = c.platform * np.array([np.sin(.7 * t), np.cos(.5 * t)])
        return j + p
    def observe(s, cam, world_targets, t):
        c = s.c; w, h = c.res
        off = s.offsets(t); tl = np.array(cam) - [w / 2, h / 2] + off
        img = np.full((h, w), 25, np.float32) + stars_layer(s.st, s.sv, tl, (h, w))
        truth = []
        for p in world_targets:
            u = p - tl
            if 0 <= u[0] < w and 0 <= u[1] < h: target_patch(img, u[0], u[1], c)
            truth.append(u)
        img = weather(img, c, s.rng)
        return apply_noise(np.clip(img, 0, 255), c, s.rng), truth   # truth in image coords

def render_world(c, p, rng):
    """Full-screen noisy frame (used to export benchmark-style .mp4)."""
    if not hasattr(c, '_st'): object.__setattr__(c, '_st', make_stars(c, np.random.default_rng(c.seed + 1000)))
    img = np.full((c.screen, c.screen), 25, np.float32) + stars_layer(*c._st, np.zeros(2), (c.screen, c.screen)); target_patch(img, p[0], p[1], c)
    return apply_noise(np.clip(weather(img, c, rng), 0, 255), c, rng)
