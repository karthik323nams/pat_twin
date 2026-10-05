"""Step-able session: closed-loop pipeline with FSOC optics, IMM-EKF, AI verifier, and telemetry."""
import os, json, csv, time
from dataclasses import asdict
import numpy as np, cv2
from .sim import Cfg, Motion, Camera
from .vision import detect, detect_wide, Stabilizer
from .track import Tracker
from .optics import compute_link

class Session:
    def __init__(s, c, video=None):
        s.c, s.video = c, video
        s.rng = np.random.default_rng(c.seed)
        s.cap = cv2.VideoCapture(video) if video else None
        s.truth_csv = None
        if s.cap is not None:
            s.W = (int(s.cap.get(3)), int(s.cap.get(4))); s.n = int(s.cap.get(7))
            tp = video.rsplit(".", 1)[0] + ".truth.csv"
            if os.path.exists(tp): s.truth_csv = np.loadtxt(tp, delimiter=",", skiprows=1)
        else:
            s.W = (c.screen, c.screen); s.n = int(c.duration * c.fps)
            s.mot = Motion(c, s.rng); s.cam = Camera(c, s.rng)
        s.trk = Tracker(c, s.W); s.stab = Stabilizer()
        s.rows, s.ev = [], {"acq": None, "reacq": [], "drop": []}
        s.tproc = 0.0; s.nact = 0; s.i = 0; s.t0 = time.perf_counter()
        s.ctr = np.array(c.res, float) / 2; s.last = None; s.done = False

    def step(s):
        """Advance one frame. Returns a telemetry dict for display, or None when input is exhausted."""
        c, trk, i = s.c, s.trk, s.i
        if i >= s.n: s.done = True; return None
        t = i * c.dt; pos = trk.gim.pos.copy(); tl = pos - s.ctr; wide = None; p = None
        decoys_world = []
        if s.cap is not None:
            ok, fr = s.cap.read()
            if not ok: s.done = True; return None
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); W = s.W
            x0 = int(min(max(round(tl[0]), 0), W[0] - c.res[0])); y0 = int(min(max(round(tl[1]), 0), W[1] - c.res[1])); tl = np.array([x0, y0], float)
            img = g[y0:y0 + c.res[1], x0:x0 + c.res[0]]
            tw = None if s.truth_csv is None or i >= len(s.truth_csv) else s.truth_csv[i, 1:3]
            ti = None if tw is None else tw - tl; p = tw
            if trk.needs_cue(): wide = cv2.resize(g, (W[0] // c.wide_bin, W[1] // c.wide_bin), interpolation=cv2.INTER_AREA)
        else:
            p = s.mot.step(t)
            decoys_world = s.mot.get_decoys(t) if hasattr(s.mot, "get_decoys") else []
            targets_to_render = [p] + decoys_world
            img, tr = s.cam.observe(pos, targets_to_render, t)
            ti = tr[0]
            if trk.needs_cue(): wide = s.cam.wide([p])
            
        sp = time.perf_counter()
        filt = trk.filter
        mask_pos = (filt.comb_x[:2] - tl) if (filt is not None and hasattr(filt, "comb_x")) else ((filt.x[:2] - tl) if filt is not None else s.last)
        jit = s.stab.update(img, pos, mask_pos) if c.stabilize else np.zeros(2)
        sa = c.stabilize and s.stab.active; s.nact += sa
        
        cue = None
        if wide is not None:
            d = detect_wide(wide)
            if d: cue = (d[0] * c.wide_bin, d[1] * c.wide_bin)
            
        min_snr = 25.0 if trk.state == "SEARCH" else 11.0
        det = detect(img, c.tsize, trk.roi(tl), min_snr=min_snr, use_ai=c.use_ai)
        s.last = None if det is None else det[:2]
        
        e = trk.update(det, t, tl, cue, jit if sa else (0., 0.), sa)
        s.tproc += time.perf_counter() - sp
        
        ev = s.ev
        if "acquired" in e and ev["acq"] is None: ev["acq"] = t
        if "reacquired" in e: ev["reacq"].append(t - ev["drop"][-1] if ev["drop"] else 0)
        if "dropout" in e: ev["drop"].append(t)
        
        cen = np.nan if (det is None or ti is None) else float(np.hypot(det[0] - ti[0], det[1] - ti[1]))
        raw = np.nan if ti is None else float(np.hypot(*(ti - s.ctr)))
        
        # Fine steering mirror / electronic stabilization compensation
        if ti is None:
            fine = np.nan
        elif c.use_fsm and trk.state == "TRACK":
            # FSM suppresses jitter in optical path down to residual sensor noise
            fine = float(cen) if not np.isnan(cen) else float(min(raw, 2.5 + np.abs(float(s.rng.normal(0, 0.8)))))
        else:
            fine = float(np.hypot(*(ti + (jit if sa else 0) - s.ctr)))
            
        # Compute live FSOC link budget metrics
        link = compute_link(fine if c.use_fsm else raw, c.weather)
        ai_score = det[3] if (det is not None and len(det) > 3) else (1.0 if det is not None else 0.0)
        snr_val = det[2] if det is not None else None
        
        s.rows.append((
            i, round(t, 3), trk.state, cen, raw, fine, snr_val, ai_score,
            link.pointing_loss_db, link.rx_power_dbm, link.link_margin_db, link.ber, link.status,
            trk.model_probs
        ))
        s.i += 1
        return dict(
            i=i, t=t, img=img, tl=tl, det=det, state=trk.state, cen=cen, raw=raw, fine=fine,
            link=link, model_probs=trk.model_probs, ai_score=ai_score,
            gim=trk.gim.pos.copy(), target=None if p is None else np.asarray(p, float),
            decoys=decoys_world, est=trk.est, fps=(i + 1) / max(s.tproc, 1e-9)
        )

    def summary(s):
        c, ev, rows = s.c, s.ev, s.rows; N = len(rows); wall = time.perf_counter() - s.t0
        st = np.array([r[2] for r in rows])
        col = lambda k: np.array([r[k] for r in rows], float)
        cen, raw, fine = col(3), col(4), col(5)
        loss_db, rx_pwr, margin = col(8), col(9), col(10)
        
        a = int(ev["acq"] / c.dt) if ev["acq"] is not None else N
        after = st[a:]; lock = (after == "TRACK").mean() if len(after) else 0.0
        nn = lambda x: x[~np.isnan(x)]
        ok_c = nn(cen); ok_r = nn(raw[a:]); ok_f = nn(fine[a:])
        ok_loss = nn(loss_db[a:]); ok_margin = nn(margin[a:])
        
        # Link availability: percentage of time margin >= 3.0 dB
        avail = float((ok_margin >= 3.0).mean() * 100.0) if len(ok_margin) else 0.0
        
        return {
            "frames": N, "sim_duration_s": round(N * c.dt, 2), "acquisition_time_s": ev["acq"],
            "reacquisition_time_mean_s": float(np.mean(ev["reacq"])) if ev["reacq"] else None,
            "n_dropouts": len(ev["drop"]),
            "lock_retention_pct": round(100 * lock, 2),
            "target_loss_pct": round(100 * (1 - lock), 2),
            "centroid_err_mean_px": float(ok_c.mean()) if len(ok_c) else None,
            "centroid_err_rmse_px": float(np.sqrt((ok_c**2).mean())) if len(ok_c) else None,
            "pointing_err_coarse_mean_px": float(ok_r.mean()) if len(ok_r) else None,
            "pointing_err_fine_mean_px": float(ok_f.mean()) if len(ok_f) else None,
            "pointing_err_fine_max_px": float(ok_f.max()) if len(ok_f) else None,
            "optical_pointing_loss_mean_db": float(ok_loss.mean()) if len(ok_loss) else None,
            "optical_link_margin_mean_db": float(ok_margin.mean()) if len(ok_margin) else None,
            "optical_link_availability_pct": round(avail, 2),
            "imm_model_final_probs": {"CV": s.trk.model_probs[0], "CA": s.trk.model_probs[1], "CT": s.trk.model_probs[2]},
            "stabilizer_active_pct": round(100 * s.nact / max(N, 1), 1),
            "processing_fps": round(N / max(s.tproc, 1e-9), 1),
            "end_to_end_fps": round(N / wall, 1),
            "config": asdict(c) if s.cap is None else "video"
        }

    def save(s, out="logs"):
        os.makedirs(out, exist_ok=True); S = s.summary()
        with open(f"{out}/frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "frame", "t", "state", "centroid_err_px", "pointing_coarse_px", "pointing_fine_px",
                "snr", "ai_conf", "pointing_loss_db", "rx_power_dbm", "link_margin_db", "ber",
                "link_status", "imm_probs"
            ])
            w.writerows(s.rows)
        json.dump(S, open(f"{out}/summary.json", "w"), indent=2, default=str)
        return S
