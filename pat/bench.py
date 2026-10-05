"""Benchmark: every motion x weather x noise combination, checked against the SIH spec limits.
   python -m pat.bench [--quick] [--duration 15] [--seeds 1] [--no-cue]"""
import argparse, csv, itertools
from .sim import Cfg
from .run import run

LIM = dict(acq=2.0, pt=10.0, loss=5.0, reacq=1.0, fps=20.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--duration", type=float, default=15)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--no-cue", action="store_true")
    ap.add_argument("--out", default="logs/bench")
    a = ap.parse_args()
    
    motions = ["straight", "circular", "figure8", "random", "spiral", "sinusoidal", "leo"]
    weathers = ["clear", "haze", "fog", "rain", "lowlight"]
    noises = [("sp", "gauss", "poisson")]
    
    if a.quick:
        motions = ["figure8", "random", "spiral", "leo"]
        weathers = ["clear", "fog"]
        noises = [("sp", "gauss", "poisson")]
        
    rows = []
    f = lambda v: "-" if v is None else f"{v:.2f}"
    
    print(f"{'motion':10s}{'weather':9s}{'noise':6s} {'acq':>6s}{'loss%':>7s}{'cent':>6s}{'pt_fine':>8s}{'pt_crs':>7s}{'margin':>7s}{'fps':>7s}  spec")
    
    for m, w, n in itertools.product(motions, weathers, noises):
        for sd in range(1, a.seeds + 1):
            cfg = Cfg(
                motion=m, weather=w, noise=n, sigma=20, jitter=8, duration=a.duration,
                seed=sd, wide_cue=not a.no_cue, use_imm=True, use_ai=True, use_fsm=True
            )
            S = run(cfg, out=a.out)
            acq = S["acquisition_time_s"]
            pt_fine = S.get("pointing_err_fine_mean_px")
            pt_coarse = S.get("pointing_err_coarse_mean_px")
            pt = pt_fine if pt_fine is not None else pt_coarse
            ra = S["reacquisition_time_mean_s"]
            margin = S.get("optical_link_margin_mean_db")
            
            ok = dict(
                acq=acq is not None and acq <= LIM["acq"],
                pt=pt is not None and pt <= LIM["pt"],
                loss=S["target_loss_pct"] < LIM["loss"],
                reacq=ra is None or ra <= LIM["reacq"],
                fps=S["processing_fps"] >= LIM["fps"]
            )
            flag = "".join("." if v else k[0].upper() for k, v in ok.items())
            
            print(f"{m:10s}{w:9s}{len(n):<6d} {f(acq):>6s}{S['target_loss_pct']:7.1f}{f(S['centroid_err_mean_px']):>6s}{f(pt_fine):>8s}{f(pt_coarse):>7s}{f(margin):>7s}{S['processing_fps']:7.0f}  {flag}")
            rows.append([m, w, "+".join(n), sd, acq, S["target_loss_pct"], ra, S["centroid_err_mean_px"], pt_fine, pt_coarse, margin, S["processing_fps"], flag])
            
    with open(f"{a.out}/bench.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["motion", "weather", "noise", "seed", "acq_s", "loss_pct", "reacq_s", "centroid_px", "pointing_fine_px", "pointing_coarse_px", "margin_db", "fps", "fails"])
        w.writerows(rows)
    print(f"\nflags: A=acquisition>2s P=pointing>10px L=loss>=5% R=re-acq>1s F=fps<20  ('.' = pass)   saved {a.out}/bench.csv")

if __name__ == "__main__": main()
