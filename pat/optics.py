"""Free-Space Optical Communications (FSOC) Link Budget & Optical Physics Engine.

Translates tracking error into received optical power, pointing loss, link margin, and BER.
"""
import math
from dataclasses import dataclass

@dataclass
class OpticalParams:
    wavelength_nm: float = 1550.0       # Standard C-band telecom laser
    tx_power_dbm: float = 30.0          # +30 dBm (1.0 W optical output)
    beam_waist_px: float = 18.0         # 1/e^2 beam radius in camera pixel equivalent
    rx_sensitivity_dbm: float = -38.0   # High-sensitivity APD receiver (-38 dBm)
    geom_loss_db: float = 56.0          # Space path loss (e.g. 15 km link)
    optics_efficiency_db: float = 3.5   # Internal optics & filter transmission loss

# Atmospheric extinction per weather condition (dB)
WEATHER_EXTINCTION_DB = {
    "clear": 1.5,
    "haze": 4.0,
    "fog": 18.0,
    "rain": 8.5,
    "lowlight": 2.0
}

@dataclass
class LinkState:
    pointing_err_px: float
    pointing_loss_db: float
    rx_power_dbm: float
    rx_power_uw: float
    link_margin_db: float
    ber: float
    status: str       # "LOCKED", "MARGINAL", "OUTAGE"

def compute_link(err_px: float, weather: str = "clear", p: OpticalParams = None) -> LinkState:
    """Computes instant optical link performance given pointing error in pixels."""
    if p is None:
        p = OpticalParams()
    
    if err_px is None or math.isnan(err_px):
        return LinkState(
            pointing_err_px=float('nan'),
            pointing_loss_db=-40.0,
            rx_power_dbm=-80.0,
            rx_power_uw=0.0,
            link_margin_db=-42.0,
            ber=0.5,
            status="OUTAGE"
        )
    
    # Pointing loss: Gaussian beam attenuation L_p = exp(-2 * (theta / w)^2)
    # In dB: L_p_dB = -4.3429 * 2 * (theta / w)^2 = -8.6858 * (err / waist)^2
    ratio = max(0.0, float(err_px)) / max(p.beam_waist_px, 1e-3)
    loss_db = 8.6858 * (ratio ** 2)
    loss_db = min(loss_db, 60.0)   # Cap max loss
    
    atm_loss = WEATHER_EXTINCTION_DB.get(weather, 2.0)
    
    # Received optical power (dBm)
    rx_power_dbm = p.tx_power_dbm - p.geom_loss_db - p.optics_efficiency_db - atm_loss - loss_db
    
    # In microwatts (uW): P_mW = 10^(dBm / 10), P_uW = P_mW * 1000
    rx_power_uw = max(1e-9, 10.0 ** ((rx_power_dbm + 30.0) / 10.0) * 1e3)
    
    # Link margin above receiver sensitivity
    margin_db = rx_power_dbm - p.rx_sensitivity_dbm
    
    # Bit Error Rate (BER) estimate for OOK / optical heterodyne:
    # Q factor relates to electrical SNR (proportional to optical power squared for direct detection)
    if margin_db > 12.0:
        ber = 1.0e-12
    elif margin_db < -3.0:
        ber = 0.5
    else:
        # Logistic / Q-function approximation: Q = 10^(margin / 20) * 3.5
        q = max(0.0, (10.0 ** (margin_db / 20.0)) * 2.8)
        # Complementary error function approx: 0.5 * erfc(q / sqrt(2))
        try:
            ber = max(1.0e-12, min(0.5, 0.5 * math.erfc(q / 1.41421356)))
        except Exception:
            ber = 0.5
            
    if margin_db >= 3.0:
        status = "LOCKED"
    elif margin_db >= 0.0:
        status = "MARGINAL"
    else:
        status = "OUTAGE"
        
    return LinkState(
        pointing_err_px=round(float(err_px), 2),
        pointing_loss_db=round(loss_db, 2),
        rx_power_dbm=round(rx_power_dbm, 2),
        rx_power_uw=round(rx_power_uw, 4),
        link_margin_db=round(margin_db, 2),
        ber=ber,
        status=status
    )
