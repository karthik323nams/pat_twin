"""Kalman & IMM-EKF trackers, lock state machine, dual-stage controller and gimbal model."""
import numpy as np

class KF:
    """Standard Constant-Velocity Kalman Filter (fallback baseline)."""
    def __init__(s, z, dt, q=400.0, r=2.0):
        s.x = np.array([z[0], z[1], 0, 0.], dtype=float)
        s.P = np.diag([r**2, r**2, 400., 400.])
        s.F = np.eye(4); s.F[0, 2] = s.F[1, 3] = dt
        G = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        s.Q = q**2 * G @ G.T; s.H = np.eye(2, 4); s.r = r
    def predict(s):
        s.x = s.F @ s.x; s.P = s.F @ s.P @ s.F.T + s.Q
        return s.x[:2]
    def update(s, z, r=None):
        R = np.eye(2) * (r or s.r)**2; S = s.H @ s.P @ s.H.T + R
        K = s.P @ s.H.T @ np.linalg.inv(S)
        s.x = s.x + K @ (np.asarray(z) - s.H @ s.x)
        s.P = (np.eye(4) - K @ s.H) @ s.P
    @property
    def unc(s): return float(np.sqrt(np.trace(s.P[:2, :2])))
    @property
    def vel(s): return s.x[2:4]


class IMMFilter:
    """Interacting Multiple Model (IMM) Filter with 3 models:
       - Model 0: Constant Velocity (CV) - low process noise cruising
       - Model 1: Constant Acceleration (CA) - maneuver & thruster bursts
       - Model 2: Coordinated Turn (CT) - circular / curved trajectory
    """
    def __init__(s, z, dt, r=2.0):
        s.dt = dt
        s.models = ["CV", "CA", "CT"]
        s.M = 3
        # Mode transition probability matrix
        s.Pi = np.array([
            [0.86, 0.08, 0.06],
            [0.10, 0.82, 0.08],
            [0.06, 0.10, 0.84]
        ], dtype=float)
        s.mu = np.array([0.60, 0.20, 0.20], dtype=float)
        
        # State: [x, y, vx, vy, ax, ay] (6-dim)
        s.x = [np.array([z[0], z[1], 0., 0., 0., 0.], dtype=float) for _ in range(s.M)]
        s.P = [np.diag([r**2, r**2, 400., 400., 100., 100.]) for _ in range(s.M)]
        
        # Measurement matrix H (2x6)
        s.H = np.zeros((2, 6), dtype=float)
        s.H[0, 0] = s.H[1, 1] = 1.0
        s.r = r
        
        # Model 0: CV Transition Matrix
        s.F_cv = np.eye(6, dtype=float)
        s.F_cv[0, 2] = s.F_cv[1, 3] = dt
        G_cv = np.array([[dt**2/2, 0], [0, dt**2/2], [dt, 0], [0, dt], [0, 0], [0, 0]], dtype=float)
        s.Q_cv = (50.0)**2 * G_cv @ G_cv.T
        
        # Model 1: CA Transition Matrix
        s.F_ca = np.eye(6, dtype=float)
        s.F_ca[0, 2] = s.F_ca[1, 3] = dt
        s.F_ca[0, 4] = s.F_ca[1, 5] = 0.5 * dt**2
        s.F_ca[2, 4] = s.F_ca[3, 5] = dt
        G_ca = np.array([[dt**2/2, 0], [0, dt**2/2], [dt, 0], [0, dt], [1, 0], [0, 1]], dtype=float)
        s.Q_ca = (140.0)**2 * G_ca @ G_ca.T
        
        # Model 2: CT Transition Matrix (turn rate approx 0.35 rad/s)
        w = 0.35
        sin_w = np.sin(w * dt); cos_w = np.cos(w * dt)
        s.F_ct = np.eye(6, dtype=float)
        s.F_ct[0, 0] = 1.0; s.F_ct[0, 2] = sin_w / w; s.F_ct[0, 3] = -(1.0 - cos_w) / w
        s.F_ct[1, 1] = 1.0; s.F_ct[1, 2] = (1.0 - cos_w) / w; s.F_ct[1, 3] = sin_w / w
        s.F_ct[2, 2] = cos_w; s.F_ct[2, 3] = -sin_w
        s.F_ct[3, 2] = sin_w; s.F_ct[3, 3] = cos_w
        s.Q_ct = (80.0)**2 * G_cv @ G_cv.T
        
        s.F = [s.F_cv, s.F_ca, s.F_ct]
        s.Q = [s.Q_cv, s.Q_ca, s.Q_ct]
        
        # Combined output
        s.comb_x = s.x[0].copy()
        s.comb_P = s.P[0].copy()

    def predict(s):
        # 1. Interaction / Mixing
        c_bar = s.Pi.T @ s.mu                                    # (3,)
        c_bar = np.maximum(c_bar, 1e-12)
        mu_mix = (s.Pi * s.mu[:, np.newaxis]) / c_bar[np.newaxis, :] # mu_mix[i, j] = P(M_i at k-1 | M_j at k)
        
        x_mixed = []
        P_mixed = []
        for j in range(s.M):
            xj = np.zeros(6, dtype=float)
            for i in range(s.M):
                xj += mu_mix[i, j] * s.x[i]
            x_mixed.append(xj)
            
            Pj = np.zeros((6, 6), dtype=float)
            for i in range(s.M):
                dx = (s.x[i] - xj)[:, np.newaxis]
                Pj += mu_mix[i, j] * (s.P[i] + dx @ dx.T)
            P_mixed.append(Pj)
            
        # 2. Model-conditional Prediction
        for j in range(s.M):
            s.x[j] = s.F[j] @ x_mixed[j]
            s.P[j] = s.F[j] @ P_mixed[j] @ s.F[j].T + s.Q[j]
            
        s._combine()
        return s.comb_x[:2]

    def update(s, z, r=None):
        R = np.eye(2) * (r or s.r)**2
        L = np.zeros(s.M, dtype=float)
        
        for j in range(s.M):
            nu = np.asarray(z) - s.H @ s.x[j]
            S = s.H @ s.P[j] @ s.H.T + R
            inv_S = np.linalg.inv(S)
            K = s.P[j] @ s.H.T @ inv_S
            s.x[j] = s.x[j] + K @ nu
            s.P[j] = (np.eye(6) - K @ s.H) @ s.P[j]
            
            # Gaussian likelihood
            det_S = max(np.linalg.det(S), 1e-12)
            d2 = float(nu @ inv_S @ nu)
            L[j] = np.exp(-0.5 * min(d2, 50.0)) / (2.0 * np.pi * np.sqrt(det_S)) + 1e-12
            
        # Mode probability update
        c_bar = s.Pi.T @ s.mu
        mu_unnorm = c_bar * L
        s.mu = mu_unnorm / max(mu_unnorm.sum(), 1e-12)
        s._combine()

    def _combine(s):
        s.comb_x = np.zeros(6, dtype=float)
        for j in range(s.M):
            s.comb_x += s.mu[j] * s.x[j]
        s.comb_P = np.zeros((6, 6), dtype=float)
        for j in range(s.M):
            dx = (s.x[j] - s.comb_x)[:, np.newaxis]
            s.comb_P += s.mu[j] * (s.P[j] + dx @ dx.T)

    @property
    def x_comb(s): return s.comb_x
    @property
    def unc(s): return float(np.sqrt(np.trace(s.comb_P[:2, :2])))
    @property
    def vel(s): return s.comb_x[2:4]
    @property
    def model_probs(s): return tuple(round(float(p), 3) for p in s.mu)


class Gimbal:
    def __init__(s, c, world):
        s.c, s.lim, s.W = c, c.max_rate * c.ppd, world
        s.pos = np.array(c.res, float) / 2 * 0 + world / 2; s.v = np.zeros(2)
    def step(s, cmd):
        n = np.linalg.norm(cmd)
        if n > s.lim: cmd = cmd * s.lim / n                          # rate saturation
        s.v += 0.7 * (cmd - s.v)                                    # first-order actuator lag
        s.pos = np.clip(s.pos + s.v * s.c.dt, [s.c.res[0] / 2, s.c.res[1] / 2],
                        [s.W[0] - s.c.res[0] / 2, s.W[1] - s.c.res[1] / 2])


class Scan:
    """Boustrophedon search over the whole screen."""
    def __init__(s, c, W):
        xs = [c.res[0] / 2, W[0] - c.res[0] / 2]
        ys = np.arange(c.res[1] / 2, W[1] - c.res[1] / 2 + 1, c.res[1] * .8)
        s.pts = [(xs[j], y) for i, y in enumerate(ys) for j in ((0, 1) if i % 2 == 0 else (1, 0))]
        s.i = 0
    def cmd(s, pos, gain=6.0):
        tgt = np.array(s.pts[s.i]); e = tgt - pos
        if np.linalg.norm(e) < 25: s.i = (s.i + 1) % len(s.pts)
        return e * gain


class Tracker:
    SEARCH, TRACK, COAST = "SEARCH", "TRACK", "COAST"
    CTRL, K, LA = "dead", 1.0, 1.0   # deadbeat lag-inverting controller
    def __init__(s, c, world):
        s.c, s.W = c, world; s.state = s.SEARCH; s.filter = None; s.hits = s.miss = 0; s.cue = None; s.cue_age = 99
        s.scan = Scan(c, world); s.gim = Gimbal(c, np.array(world, float)); s.est = None
        s.model_probs = (1.0, 0.0, 0.0)
        
    def needs_cue(s): return s.c.wide_cue and (s.state == s.SEARCH or (s.state == s.COAST and s.miss > 6))
    
    def roi(s, tl):
        if s.state == s.SEARCH or s.filter is None: return None
        pos = s.filter.comb_x[:2] if hasattr(s.filter, "comb_x") else s.filter.x[:2]
        u = pos - tl
        return u[0], u[1], 90 + 6 * min(s.miss, 40) + 4 * s.filter.unc
        
    def update(s, det, t, tl, cue=None, jit=(0., 0.), stab=False):
        """det=(x,y,snr,...) image coords or None; tl=viewport top-left (world px); cue=(x,y) world px."""
        c = s.c; ev = {}; tl = np.asarray(tl, float)
        z = None if det is None else tl + np.asarray(jit) + det[:2]
        
        if cue is not None: s.cue, s.cue_age = np.asarray(cue, float), 0
        else: s.cue_age += 1
        
        # Adaptive measurement noise standard deviation
        r_adaptive = 2.0
        if det is not None:
            snr = det[2] if len(det) > 2 else 25.0
            r_adaptive = max(0.5, 30.0 / max(snr, 1.0)) + (0.1 if stab else 0.25 * c.jitter)
            
        if s.state == s.SEARCH:
            s.hits = s.hits + 1 if z is not None else 0
            if s.hits >= (2 if s.cue_age < 40 else 3):
                if c.use_imm:
                    s.filter = IMMFilter(z, c.dt, r=r_adaptive)
                else:
                    s.filter = KF(z, c.dt, r=r_adaptive)
                s.state = s.TRACK; s.miss = 0; ev["acquired"] = t
        else:
            pred = s.filter.predict()
            gate = 90 + 5 * min(s.miss, 60) + 3 * s.filter.unc
            if z is not None and np.linalg.norm(z - pred) < gate:
                s.filter.update(z, r=r_adaptive)
                if s.state == s.COAST: ev["reacquired"] = t
                s.state, s.miss = s.TRACK, 0
            else:
                s.miss += 1
                if s.miss > 3: s.state = s.COAST
                if s.miss > 90: s.state = s.SEARCH; s.hits = 0; s.filter = None; ev["lost"] = t
            if s.state != s.SEARCH and s.miss == 4: ev["dropout"] = t
            
        if s.filter is not None:
            cur_pos = s.filter.comb_x[:2] if hasattr(s.filter, "comb_x") else s.filter.x[:2]
            cur_vel = s.filter.vel
            s.model_probs = s.filter.model_probs if hasattr(s.filter, "model_probs") else (1.0, 0.0, 0.0)
            aim = cur_pos + cur_vel * c.dt; ff = cur_vel; s.est = cur_pos.copy()
            
            if s.state == s.COAST and s.miss > 6 and s.cue_age < 10 and np.linalg.norm(s.cue - aim) < 500:
                aim, ff = s.cue, 0 * ff
                
            if s.CTRL == 'dead' and s.state == s.TRACK:
                des = (aim + ff * c.dt * (s.LA - 1) - s.gim.pos) * s.K / c.dt
                cmd = s.gim.v + (des - s.gim.v) / 0.7
            else:
                cmd = 14.0 * (aim - s.gim.pos) + ff
        elif s.cue_age < 40:
            cmd = 8.0 * (s.cue - s.gim.pos); s.est = None; s.model_probs = (1.0, 0.0, 0.0)
        else:
            cmd = s.scan.cmd(s.gim.pos); s.est = None; s.model_probs = (1.0, 0.0, 0.0)
            
        s.gim.step(cmd)
        return ev
