"""Preprocessing, blob detection, AI candidate verification, and sub-pixel centroiding."""
import numpy as np, cv2
from .ai_verifier import AIBeaconVerifier

_DEFAULT_VERIFIER = None

def get_verifier():
    global _DEFAULT_VERIFIER
    if _DEFAULT_VERIFIER is None:
        _DEFAULT_VERIFIER = AIBeaconVerifier()
    return _DEFAULT_VERIFIER

def detect(img, tsize, roi=None, min_snr=25.0, use_ai=False):
    """Returns (x, y, snr, ai_score) in image coords, or None. roi=(cx,cy,half) restricts search."""
    x0 = y0 = 0
    if roi is not None:
        cx, cy, hf = map(int, roi)
        x0, y0 = max(0, cx - hf), max(0, cy - hf)
        img = img[y0:min(img.shape[0], cy + hf), x0:min(img.shape[1], cx + hf)]
        if img.shape[0] < tsize * 2 or img.shape[1] < tsize * 2: return None
        
    im = cv2.medianBlur(img, 3)                                  # kills salt & pepper
    f = cv2.blur(im.astype(np.float32), (tsize, tsize))          # matched filter (square)
    r = f - cv2.blur(f, (51, 51))                                # local background removal
    sig = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-3
    _, mx, _, (px, py) = cv2.minMaxLoc(r)
    snr = mx / sig
    if snr < min_snr or mx < 6: return None
    
    # Sub-pixel centroiding around candidate peak
    h = tsize
    xa, xb = max(0, px - h), min(im.shape[1], px + h + 1); ya, yb = max(0, py - h), min(im.shape[0], py + h + 1)
    win = im[ya:yb, xa:xb].astype(np.float32)
    wgt = np.clip(win - (win.min() + 0.5 * (win.max() - win.min())), 0, None)   # half-max threshold
    if wgt.sum() <= 0:
        cx_sub, cy_sub = px, py
    else:
        gy, gx = np.mgrid[ya:yb, xa:xb]
        cx_sub = float((wgt * gx).sum() / wgt.sum())
        cy_sub = float((wgt * gy).sum() / wgt.sum())
        
    ai_score = 1.0
    if use_ai:
        ver = get_verifier()
        ai_score = ver.verify(im, cx_sub, cy_sub, patch_size=32)
        # Reject false alarms / hot pixels with low optical spot score
        if ai_score < 0.22:
            return None
            
    return x0 + cx_sub + .5, y0 + cy_sub + .5, snr, ai_score

def detect_wide(img, min_snr=9.0):
    """Coarse detection on the binned full-screen image. Returns (x, y, snr) in binned px or None."""
    f = cv2.blur(img.astype(np.float32), (2, 2)); r = f - cv2.blur(f, (15, 15))
    sig = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-3
    _, mx, _, (px, py) = cv2.minMaxLoc(r)
    return None if mx / sig < min_snr else (px + .5, py + .5, mx / sig)

class Stabilizer:
    """Estimates image shift (jitter + platform vibration) from the background.
    
    Uses frame-to-frame phase correlation compensated by known gimbal slewing,
    preventing long-range keyframe degradation.
    """
    def __init__(s, min_resp=0.02):
        s.prev_frame = None
        s.prev_cam = None
        s.min_resp = min_resp
        s.active = False
        s.win = None
        s.jitter_estimate = np.zeros(2)
        
    def update(s, img, cam, mask=None):
        cur = cv2.GaussianBlur(cv2.medianBlur(img, 3).astype(np.float32), (0, 0), 1.2)
        cam = np.asarray(cam, float)
        
        if mask is not None:
            # Mask out the beacon to prevent target motion from contaminating background shift
            cv2.circle(cur, (int(mask[0]), int(mask[1])), 22, float(np.median(cur)), -1)
            
        if s.prev_frame is None or s.win is None or s.win.shape != cur.shape:
            s.prev_frame = cur.copy()
            s.prev_cam = cam.copy()
            s.win = cv2.createHanningWindow(cur.shape[::-1], cv2.CV_32F)
            s.active = False
            return np.zeros(2)
            
        dcam = cam - s.prev_cam  # Mechanical motion commanded
        (dx, dy), resp = cv2.phaseCorrelate(s.prev_frame, cur, s.win)
        s.active = bool(resp >= s.min_resp)
        
        if s.active:
            # Shift measured in image = -(dcam + jitter_diff)
            # Therefore: instantaneous jitter delta = -shift - dcam
            d_jitter = -np.array([dx, dy], float) - dcam
            # Limit wild outliers
            d_jitter = np.clip(d_jitter, -25.0, 25.0)
            s.jitter_estimate = 0.85 * s.jitter_estimate + 0.15 * d_jitter
        else:
            s.jitter_estimate *= 0.5   # Decay when background texture lost
            
        s.prev_frame = cur.copy()
        s.prev_cam = cam.copy()
        return s.jitter_estimate.copy()
