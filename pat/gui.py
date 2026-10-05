"""PAT-Twin GUI (PyQt6) - Physics-Faithful Digital Twin for Free-Space Optical Communications.
Run:  python -m pat.gui
"""
import sys, os, json, math
from collections import deque
from dataclasses import asdict
import numpy as np, cv2
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QPdfWriter, QPageSize, QFont, QPolygonF
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QComboBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QLabel, QFileDialog, QMessageBox, QGridLayout, QFrame)
from .sim import Cfg
from .engine import Session

PT_LIMIT = 10.0
STATE_COL = {"SEARCH": "#e0a800", "TRACK": "#2ecc71", "COAST": "#e74c3c"}
LINK_COL = {"LOCKED": "#2ecc71", "MARGINAL": "#f39c12", "OUTAGE": "#e74c3c"}

def to_qimage(img):
    img = np.ascontiguousarray(img)
    if img.ndim == 2: return QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format.Format_Grayscale8).copy()
    return QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format.Format_RGB888).copy()

class ErrorPlot(QWidget):
    def __init__(s, n=300):
        super().__init__()
        s.setMinimumHeight(140)
        s.raw = deque(maxlen=n)
        s.fine = deque(maxlen=n)
        s.cen = deque(maxlen=n)
        
    def push(s, raw, fine, cen):
        s.raw.append(raw); s.fine.append(fine); s.cen.append(cen); s.update()
        
    def clear(s):
        s.raw.clear(); s.fine.clear(); s.cen.clear(); s.update()
        
    def draw(s, p, r, ymax=35.0):
        p.fillRect(r, QColor("#111116"))
        p.setPen(QColor("#aaa"))
        p.setFont(QFont("Arial", 8))
        p.drawText(r.adjusted(8, 4, 0, 0), Qt.AlignmentFlag.AlignTop,
                   "Error (px):   Yellow = Coarse Gimbal Err   Green = Fine Pointing Err (Spec <= 10px)   Cyan = Centroid Err")
        
        y = lambda v: r.bottom() - min(v, ymax) / ymax * (r.height() - 8) - 4
        # Red dashed spec limit line at 10.0 px
        p.setPen(QPen(QColor("#e74c3c"), 1, Qt.PenStyle.DashLine))
        y_lim = y(PT_LIMIT)
        p.drawLine(QPointF(r.left(), y_lim), QPointF(r.right(), y_lim))
        p.drawText(QPointF(r.right() - 75, y_lim - 2), "10 px limit")
        
        # Plot curves
        for data, col, width in ((s.raw, "#f1c40f", 1), (s.fine, "#2ecc71", 2), (s.cen, "#4dd0e1", 1)):
            p.setPen(QPen(QColor(col), width))
            seg = []; n = data.maxlen; d = list(data)
            for k, v in enumerate(d):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    if len(seg) > 1: p.drawPolyline(QPolygonF(seg))
                    seg = []
                else:
                    seg.append(QPointF(r.left() + k / max(n - 1, 1) * r.width(), y(v)))
            if len(seg) > 1: p.drawPolyline(QPolygonF(seg))

    def paintEvent(s, e):
        p = QPainter(s); s.draw(p, QRectF(s.rect())); p.end()

class MainWindow(QMainWindow):
    def __init__(s):
        super().__init__()
        s.setWindowTitle("PAT-Twin - Physics-Faithful FSOC Pointing, Acquisition & Tracking Digital Twin")
        s.sess = None; s.video = None
        s.out = os.path.join(os.path.expanduser("~"), "PAT-Twin-logs")
        s.timer = QTimer(s); s.timer.setInterval(15); s.timer.timeout.connect(s.tick)
        
        root = QWidget(); s.setCentralWidget(root)
        h = QHBoxLayout(root)
        h.addWidget(s.build_controls(), 0)
        
        mid = QVBoxLayout(); h.addLayout(mid, 1)
        
        # Top banner: Live FSOC Optical Link & IMM Status
        mid.addWidget(s.build_optics_bar())
        
        # Middle viewports: Sky Canvas + Camera Feed
        views = QHBoxLayout(); mid.addLayout(views, 1)
        s.world = QLabel(); s.world.setFixedSize(420, 420)
        s.cam = QLabel(); s.cam.setFixedSize(640, 480)
        for lab, t in ((s.world, "Sky Canvas (Screen Space & Decoys)"), (s.cam, "Optical Camera Viewport")):
            lab.setStyleSheet("background:#000; border:1px solid #333;")
            box = QVBoxLayout(); box.addWidget(QLabel(f"<b>{t}</b>")); box.addWidget(lab); box.addStretch(); views.addLayout(box)
            
        # Error plot
        s.plot = ErrorPlot(); mid.addWidget(s.plot)
        
        # Telemetry stats
        s.stats = QLabel("Ready to simulate."); s.stats.setStyleSheet("font-family:monospace;font-size:12px;color:#eee")
        mid.addWidget(s.stats)
        
        # Bottom system status bar
        s.status = QLabel("IDLE"); s.status.setStyleSheet("background:#555;color:white;font-weight:bold;padding:5px")
        mid.addWidget(s.status)

    def build_optics_bar(s):
        bar = QFrame()
        bar.setStyleSheet("background:#151821; border-radius:4px; padding:6px; border:1px solid #2a2e3d;")
        lay = QHBoxLayout(bar)
        
        s.link_badge = QLabel("OPTICAL LINK: OFF")
        s.link_badge.setStyleSheet("background:#444; color:#fff; font-weight:bold; padding:4px 8px; border-radius:3px;")
        lay.addWidget(s.link_badge)
        
        s.opt_margin = QLabel("Margin: -- dB")
        s.opt_loss = QLabel("Pointing Loss: -- dB")
        s.opt_pwr = QLabel("Rx Power: -- dBm")
        s.opt_ber = QLabel("Est. BER: --")
        s.imm_label = QLabel("IMM: CV 60% | CA 20% | CT 20%")
        s.ai_label = QLabel("AI Spot: --")
        
        for w in (s.opt_margin, s.opt_loss, s.opt_pwr, s.opt_ber, s.imm_label, s.ai_label):
            w.setStyleSheet("color:#ccc; font-size:11px; font-weight:bold; margin-left:8px;")
            lay.addWidget(w)
        lay.addStretch()
        return bar

    def build_controls(s):
        g = QGroupBox("Configuration"); f = QFormLayout(g); g.setFixedWidth(290)
        s.motion = QComboBox()
        s.motion.addItems(["straight", "circular", "figure8", "random", "spiral", "sinusoidal", "leo"])
        s.motion.setCurrentText("figure8")
        
        s.weather = QComboBox()
        s.weather.addItems(["clear", "haze", "fog", "rain", "lowlight"])
        
        s.n_sp = QCheckBox("Salt & Pepper")
        s.n_g = QCheckBox("Gaussian Noise")
        s.n_p = QCheckBox("Poisson Shot Noise")
        s.n_sp.setChecked(True); s.n_g.setChecked(True)
        
        def dsb(lo, hi, v, st=1.0):
            b = QDoubleSpinBox(); b.setRange(lo, hi); b.setValue(v); b.setSingleStep(st); return b
            
        s.sigma = dsb(0, 20, 15)
        s.jitter = dsb(0, 20, 20)
        s.speed = dsb(20, 800, 250, 10)
        s.duration = dsb(3, 300, 30)
        s.tsize = QSpinBox(); s.tsize.setRange(5, 20); s.tsize.setValue(10)
        s.decoys = QSpinBox(); s.decoys.setRange(0, 5); s.decoys.setValue(0)
        s.seed = QSpinBox(); s.seed.setRange(0, 99999); s.seed.setValue(1)
        s.screen = QSpinBox(); s.screen.setRange(800, 4000); s.screen.setSingleStep(100); s.screen.setValue(2000)
        
        s.cue = QCheckBox("Wide-area cue sensor"); s.cue.setChecked(True)
        s.imm = QCheckBox("IMM Filter (CV, CA, CT)"); s.imm.setChecked(True)
        s.ai = QCheckBox("AI CNN Spot Verifier"); s.ai.setChecked(True)
        s.fsm = QCheckBox("Fine Steering Mirror (FSM)"); s.fsm.setChecked(True)
        s.stab = QCheckBox("Electronic Stabilizer (EIS)")
        
        for k, w in (
            ("Target Motion", s.motion), ("Weather", s.weather), ("Noise Types", s.n_sp),
            ("", s.n_g), (" ", s.n_p), ("Gauss Sigma", s.sigma), ("Jitter ±px", s.jitter),
            ("Speed px/s", s.speed), ("Beacon Size px", s.tsize), ("Decoys Count", s.decoys),
            ("Duration s", s.duration), ("Screen px", s.screen), ("Seed", s.seed),
            ("", s.cue), (" ", s.imm), ("  ", s.ai), ("   ", s.fsm), ("    ", s.stab)
        ): f.addRow(k, w)
        
        s.b_start = QPushButton("Start Simulation")
        s.b_pause = QPushButton("Pause")
        s.b_stop = QPushButton("Stop")
        s.b_video = QPushButton("Load Video (.mp4)…")
        s.b_save = QPushButton("Save Scenario JSON…")
        s.b_load = QPushButton("Load Scenario JSON…")
        s.b_exp = QPushButton("Export Performance Report…")
        
        s.b_start.clicked.connect(s.start)
        s.b_pause.clicked.connect(s.pause)
        s.b_stop.clicked.connect(s.stop)
        s.b_video.clicked.connect(s.pick_video)
        s.b_save.clicked.connect(s.save_scn)
        s.b_load.clicked.connect(s.load_scn)
        s.b_exp.clicked.connect(s.export)
        
        s.vlabel = QLabel("Input: Synthetic Twin"); s.vlabel.setWordWrap(True)
        f.addRow(s.vlabel)
        for b in (s.b_start, s.b_pause, s.b_stop, s.b_video, s.b_save, s.b_load, s.b_exp): f.addRow(b)
        s.b_pause.setEnabled(False); s.b_stop.setEnabled(False); s.b_exp.setEnabled(False)
        return g

    def cfg(s):
        n = tuple(k for k, w in (("sp", s.n_sp), ("gauss", s.n_g), ("poisson", s.n_p)) if w.isChecked())
        return Cfg(
            motion=s.motion.currentText(), weather=s.weather.currentText(), noise=n,
            sigma=s.sigma.value(), jitter=s.jitter.value(), speed=s.speed.value(),
            tsize=s.tsize.value(), duration=s.duration.value(), screen=s.screen.value(),
            seed=s.seed.value(), wide_cue=s.cue.isChecked(), stabilize=s.stab.isChecked(),
            use_imm=s.imm.isChecked(), use_ai=s.ai.isChecked(), use_fsm=s.fsm.isChecked(),
            decoys=s.decoys.value()
        )

    def set_cfg(s, d):
        s.motion.setCurrentText(d.get("motion", "figure8"))
        s.weather.setCurrentText(d.get("weather", "clear"))
        n = d.get("noise", [])
        s.n_sp.setChecked("sp" in n); s.n_g.setChecked("gauss" in n); s.n_p.setChecked("poisson" in n)
        for w, k in (
            (s.sigma, "sigma"), (s.jitter, "jitter"), (s.speed, "speed"),
            (s.duration, "duration"), (s.tsize, "tsize"), (s.screen, "screen"),
            (s.seed, "seed"), (s.decoys, "decoys")
        ):
            if k in d: w.setValue(d[k])
        s.cue.setChecked(d.get("wide_cue", True))
        s.imm.setChecked(d.get("use_imm", True))
        s.ai.setChecked(d.get("use_ai", True))
        s.fsm.setChecked(d.get("use_fsm", True))
        s.stab.setChecked(d.get("stabilize", False))

    def running(s): return s.sess is not None and not s.sess.done
    
    def start(s):
        if s.running(): s.timer.stop()
        try:
            s.sess = Session(s.cfg(), s.video)
        except Exception as e:
            QMessageBox.critical(s, "Cannot start", str(e)); return
        s.plot.clear()
        s.b_pause.setEnabled(True); s.b_stop.setEnabled(True); s.b_exp.setEnabled(False)
        s.b_pause.setText("Pause"); s.timer.start()
        
    def pause(s):
        if s.timer.isActive(): s.timer.stop(); s.b_pause.setText("Resume")
        elif s.sess and not s.sess.done: s.timer.start(); s.b_pause.setText("Pause")
        
    def stop(s): s.finish()
    
    def finish(s):
        s.timer.stop(); s.b_pause.setEnabled(False); s.b_stop.setEnabled(False)
        if s.sess and s.sess.rows:
            s.b_exp.setEnabled(True); S = s.sess.save(s.out); s.show_summary(S)
            
    def show_summary(s, S):
        f = lambda v, u="": "n/a" if v is None else f"{v:.2f}{u}"
        s.stats.setText(
            f"FINISHED | Acq: {f(S['acquisition_time_s'], 's')} | Re-acq: {f(S['reacquisition_time_mean_s'], 's')} | "
            f"Lock: {S['lock_retention_pct']}% | Loss: {S['target_loss_pct']}%\n"
            f"Centroid RMSE: {f(S['centroid_err_rmse_px'], 'px')} | Fine Pointing Mean: {f(S.get('pointing_err_fine_mean_px'), 'px')} (Coarse: {f(S.get('pointing_err_coarse_mean_px'), 'px')})\n"
            f"Link Margin: {f(S.get('optical_link_margin_mean_db'), 'dB')} | Availability: {f(S.get('optical_link_availability_pct'), '%')} | Proc FPS: {S['processing_fps']}"
        )
        s.status.setText("COMPLETED - LOGS SAVED"); s.status.setStyleSheet("background:#2c3e50;color:white;font-weight:bold;padding:5px")

    def tick(s):
        f = s.sess.step()
        if f is None: s.finish(); return
        img = cv2.cvtColor(f["img"], cv2.COLOR_GRAY2RGB)
        
        # Render visual markers
        if f["det"]:
            cv2.circle(img, (int(f["det"][0]), int(f["det"][1])), 16, (0, 255, 0), 1)
        cv2.drawMarker(img, (img.shape[1] // 2, img.shape[0] // 2), (255, 60, 60), cv2.MARKER_CROSS, 20, 1)
        
        s.cam.setPixmap(QPixmap.fromImage(to_qimage(img)))
        s.draw_world(f)
        s.plot.push(f["raw"], f["fine"], f["cen"])
        
        # Update lock and optical link indicators
        st_col = STATE_COL.get(f["state"], "#555")
        s.status.setText(f"TRACKER STATE: {f['state']} (Target: Locked)")
        s.status.setStyleSheet(f"background:{st_col};color:white;font-weight:bold;padding:5px")
        
        link = f.get("link")
        if link:
            l_col = LINK_COL.get(link.status, "#555")
            s.link_badge.setText(f"FSOC LINK: {link.status}")
            s.link_badge.setStyleSheet(f"background:{l_col}; color:#fff; font-weight:bold; padding:4px 8px; border-radius:3px;")
            s.opt_margin.setText(f"Margin: {link.link_margin_db:+.1f} dB")
            s.opt_loss.setText(f"Loss: {link.pointing_loss_db:.1f} dB")
            s.opt_pwr.setText(f"Rx Power: {link.rx_power_dbm:.1f} dBm ({link.rx_power_uw:.1f} uW)")
            s.opt_ber.setText(f"BER: {link.ber:.1e}")
            
        probs = f.get("model_probs", (1.0, 0.0, 0.0))
        s.imm_label.setText(f"IMM: CV {int(probs[0]*100)}% | CA {int(probs[1]*100)}% | CT {int(probs[2]*100)}%")
        ai_s = f.get("ai_score")
        s.ai_label.setText(f"AI Spot: {'--' if ai_s is None else f'{ai_s:.2f}'}")
        
        n = lambda v: "  n/a" if (v is None or math.isnan(v)) else f"{v:5.1f}"
        S = s.sess
        s.stats.setText(
            f"t={f['t']:6.2f}s  frame {f['i']+1}/{S.n} | Centroid: {n(f['cen'])}px | "
            f"Fine Pointing: {n(f['fine'])}px (Coarse: {n(f['raw'])}px) | FPS: {f['fps']:.0f} | "
            f"Acq: {'n/a' if S.ev['acq'] is None else '%.2fs' % S.ev['acq']}"
        )

    def draw_world(s, f):
        W = s.sess.W; k = 420 / max(W)
        pm = QPixmap(int(W[0] * k), int(W[1] * k)); pm.fill(QColor("#0a0a14"))
        p = QPainter(pm)
        r = s.sess.c.res; g = f["gim"]
        
        # Viewport rectangle
        p.setPen(QPen(QColor("#4dd0e1"), 1))
        p.drawRect(QRectF((g[0] - r[0] / 2) * k, (g[1] - r[1] / 2) * k, r[0] * k, r[1] * k))
        
        # Decoys
        for dec in f.get("decoys", []):
            p.setPen(QPen(QColor("#e67e22"), 3))
            p.drawPoint(QPointF(dec[0] * k, dec[1] * k))
            
        # Target
        if f["target"] is not None:
            p.setPen(QPen(QColor("#ffffff"), 4))
            p.drawPoint(QPointF(f["target"][0] * k, f["target"][1] * k))
            
        # Estimated position
        if f["est"] is not None:
            p.setPen(QPen(QColor("#2ecc71"), 1))
            e = f["est"]
            p.drawEllipse(QPointF(e[0] * k, e[1] * k), 6, 6)
            
        p.end()
        s.world.setPixmap(pm)

    def pick_video(s):
        p, _ = QFileDialog.getOpenFileName(s, "Select benchmark video", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if p:
            s.video = p; s.vlabel.setText(f"Input: {os.path.basename(p)}")
            
    def save_scn(s):
        p, _ = QFileDialog.getSaveFileName(s, "Save scenario", "scenario.json", "JSON (*.json)")
        if p:
            d = asdict(s.cfg()); d["noise"] = list(d["noise"])
            json.dump(d, open(p, "w"), indent=2)
            
    def load_scn(s):
        p, _ = QFileDialog.getOpenFileName(s, "Load scenario", "", "JSON (*.json)")
        if p:
            s.set_cfg(json.load(open(p))); s.video = None; s.vlabel.setText("Input: Synthetic Twin")
            
    def export(s):
        d = QFileDialog.getExistingDirectory(s, "Export report to folder")
        if not d or not s.sess: return
        S = s.sess.save(d)
        pdf = QPdfWriter(os.path.join(d, "PAT_Twin_Report.pdf"))
        pdf.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        pdf.setResolution(96)
        
        p = QPainter(pdf)
        p.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
        y = 35
        p.drawText(30, y, "PAT-Twin: Performance & Free-Space Optics Verification Report")
        y += 24
        
        p.setFont(QFont("Courier", 9))
        p.drawText(30, y, "=" * 80); y += 14
        
        for k, v in S.items():
            if k != "config":
                val = f"{v:.3f}" if isinstance(v, float) else str(v)
                p.drawText(30, y, f"{k:<35}: {val}")
                y += 14
                
        p.drawText(30, y, "=" * 80); y += 20
        p.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        p.drawText(30, y, "Pointing & Centroid Error Trajectory (with 10 px Spec Boundary):")
        y += 10
        s.plot.draw(p, QRectF(30, y, 700, 240))
        p.end()
        s.status.setText(f"Report and logs exported to {d}")

def main():
    a = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1420, 840)
    w.show()
    sys.exit(a.exec())

if __name__ == "__main__": main()
