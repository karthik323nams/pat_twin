"""Runner: synthetic sim or video input, metrics, logs, optional live view."""
import argparse, json, os, time, csv
from dataclasses import asdict
import numpy as np, cv2
from .sim import Cfg, Motion, Camera, render_world
from .vision import detect, detect_wide, Stabilizer
from .track import Tracker

def run(c, video=None, out="logs", show=False):
    from .engine import Session
    s = Session(c, video)
    while True:
        f = s.step()
        if f is None: break
        if show:
            v = cv2.cvtColor(f["img"], cv2.COLOR_GRAY2BGR)
            if f["det"]: cv2.circle(v, (int(f["det"][0]), int(f["det"][1])), 14, (0, 255, 0), 1)
            err_disp = f.get('fine', f.get('raw', 0.0))
            lk_stat = f['link'].status if 'link' in f else ''
            pwr = f"{f['link'].rx_power_dbm:.1f}dBm" if 'link' in f else ''
            cv2.putText(v, f"{f['state']} fine_err={err_disp:.1f}px {lk_stat} {pwr}", (10, 25), 0, .6, (0, 255, 255), 2)
            cv2.imshow("PAT-Twin", v)
            if cv2.waitKey(1) == 27: break
    return s.save(out)

def export_video(c, path, secs):
    """Render a noisy full-screen .mp4 + truth sidecar (mimics benchmark-2 input)."""
    rng = np.random.default_rng(c.seed); mot = Motion(c, rng); vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), c.fps, (c.screen, c.screen))
    tr = []
    for i in range(int(secs * c.fps)):
        p = mot.step(i * c.dt); vw.write(cv2.cvtColor(render_world(c, p, rng), cv2.COLOR_GRAY2BGR)); tr.append((i, *p))
    vw.release(); np.savetxt(path.rsplit(".", 1)[0] + ".truth.csv", tr, delimiter=",", header="frame,x,y", comments="")

def main():
    ap = argparse.ArgumentParser(prog="pat")
    ap.add_argument("--motion", default="circular")
    ap.add_argument("--noise", default="sp,gauss")
    ap.add_argument("--sigma", type=float, default=15)
    ap.add_argument("--weather", default="clear")
    ap.add_argument("--jitter", type=float, default=8)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--speed", type=float, default=250)
    ap.add_argument("--tsize", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--screen", type=int, default=2000)
    ap.add_argument("--video")
    ap.add_argument("--export-video")
    ap.add_argument("--out", default="logs")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-cue", action="store_true")
    ap.add_argument("--stab", action="store_true")
    ap.add_argument("--no-imm", action="store_true", help="Use baseline single-model KF instead of IMM")
    ap.add_argument("--no-ai", action="store_true", help="Disable AI candidate verifier")
    ap.add_argument("--no-fsm", action="store_true", help="Disable Fine Steering Mirror stage")
    ap.add_argument("--decoys", type=int, default=0, help="Number of false decoy targets")
    a = ap.parse_args()
    c = Cfg(
        motion=a.motion, noise=tuple(x for x in a.noise.split(",") if x), sigma=a.sigma,
        weather=a.weather, jitter=a.jitter, duration=a.duration, speed=a.speed,
        tsize=a.tsize, seed=a.seed, screen=a.screen, wide_cue=not a.no_cue,
        stabilize=a.stab, use_imm=not a.no_imm, use_ai=not a.no_ai,
        use_fsm=not a.no_fsm, decoys=a.decoys
    )
    if a.export_video: export_video(c, a.export_video, a.duration); print("exported", a.export_video); return
    S = run(c, a.video, a.out, a.show); S.pop("config", None); print(json.dumps(S, indent=2, default=str))

if __name__ == "__main__": main()
