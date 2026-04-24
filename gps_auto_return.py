"""
GPS Auto Return Test
====================
Hardware : Raspberry Pi Zero 2W
GPS      : NEO-6M / GNSS (UART)
Driver   : L298N Dual H-Bridge

Behaviour
---------
1. Wait for a valid GPS fix → save as HOME
2. Drive in random directions for 10 seconds (wander phase)
3. Navigate back to HOME using GPS Course Over Ground (COG)
4. Stop when within HOME_THRESHOLD metres of home

Wiring (BCM numbering)
----------------------
L298N           Pi Zero 2W
------          ----------
ENA   →  GPIO12   (HW PWM)
IN1   →  GPIO5
IN2   →  GPIO6
IN3   →  GPIO13
IN4   →  GPIO19
ENB   →  GPIO26   (SW PWM)
GND   →  GND
5V    →  5V (or external supply)

NEO-6M          Pi Zero 2W
------          ----------
TX    →  GPIO15 / Pin 10 (UART RX)
RX    →  GPIO14 / Pin  8 (UART TX)
VCC   →  3.3V / Pin  1
GND   →  GND  / Pin  6

Before running
--------------
sudo raspi-config
  → Interface Options → Serial Port
      Login shell over serial? NO
      Serial port hardware enabled? YES

pip3 install pyserial pynmea2 RPi.GPIO
"""

import time
import math
import random
import signal
import sys

import serial
import pynmea2
import RPi.GPIO as GPIO

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════

# ── GPS ────────────────────────────────────────────────────
GPS_PORT       = '/dev/serial0'
GPS_BAUD       = 9600
GPS_TIMEOUT    = 2          # seconds per readline

# ── Navigation ─────────────────────────────────────────────
HOME_THRESHOLD       = 2.0   # metres  — declare "home reached"
MIN_COG_SPEED        = 0.25  # m/s     — ignore COG below this speed
HEADING_TOLERANCE    = 15    # degrees — within this → go straight
WANDER_DURATION      = 10    # seconds of random driving

# ── L298N GPIO pins (BCM) ──────────────────────────────────
ENA = 12    # Left  motor enable  (Hardware PWM — do not change)
IN1 = 5     # Left  motor forward
IN2 = 6     # Left  motor backward
IN3 = 13    # Right motor forward
IN4 = 19    # Right motor backward
ENB = 26    # Right motor enable  (Software PWM)

DRIVE_SPEED  = 65   # % duty cycle for straight driving  (0-100)
TURN_SPEED   = 55   # % duty cycle during turns          (0-100)
PWM_FREQ     = 1000 # Hz

# ═══════════════════════════════════════════════════════════
#  GPIO / MOTOR SETUP
# ═══════════════════════════════════════════════════════════

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup([ENA, IN1, IN2, IN3, IN4, ENB], GPIO.OUT)

_pwm_a = GPIO.PWM(ENA, PWM_FREQ)
_pwm_b = GPIO.PWM(ENB, PWM_FREQ)
_pwm_a.start(0)
_pwm_b.start(0)


def _set(left_fwd: bool, right_fwd: bool, left_spd: int, right_spd: int):
    GPIO.output(IN1, GPIO.HIGH if left_fwd  else GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW  if left_fwd  else GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH if right_fwd else GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW  if right_fwd else GPIO.HIGH)
    _pwm_a.ChangeDutyCycle(left_spd)
    _pwm_b.ChangeDutyCycle(right_spd)


def forward():
    print("  ↑  FORWARD")
    _set(True, True, DRIVE_SPEED, DRIVE_SPEED)


def backward():
    print("  ↓  BACKWARD")
    _set(False, False, DRIVE_SPEED, DRIVE_SPEED)


def turn_left():
    print("  ←  TURN LEFT")
    _set(False, True, TURN_SPEED, TURN_SPEED)


def turn_right():
    print("  →  TURN RIGHT")
    _set(True, False, TURN_SPEED, TURN_SPEED)


def stop():
    print("  ■  STOP")
    _pwm_a.ChangeDutyCycle(0)
    _pwm_b.ChangeDutyCycle(0)


def cleanup():
    stop()
    _pwm_a.stop()
    _pwm_b.stop()
    GPIO.cleanup()


# ═══════════════════════════════════════════════════════════
#  GPS HELPERS
# ═══════════════════════════════════════════════════════════

def read_gps(ser: serial.Serial) -> dict | None:
    """
    Read one valid GPRMC / GNRMC sentence.
    Returns dict {lat, lon, speed_ms, cog} or None on failure.
    """
    try:
        raw = ser.readline().decode('ascii', errors='replace').strip()
        if raw.startswith(('$GPRMC', '$GNRMC')):
            msg = pynmea2.parse(raw)
            if msg.status == 'A':           # 'A' = valid fix
                course = msg.true_course
                return {
                    'lat'     : msg.latitude,
                    'lon'     : msg.longitude,
                    'speed_ms': float(msg.spd_over_grnd or 0) * 0.51444,  # knots→m/s
                    'cog'     : float(course) if course else 0.0,
                }
    except Exception:
        pass
    return None


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in metres between two GPS coordinates."""
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_to(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return compass bearing (0-360°) FROM (lat1,lon1) TO (lat2,lon2)."""
    dl = math.radians(lon2 - lon1)
    x  = math.sin(dl) * math.cos(math.radians(lat2))
    y  = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
          - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dl))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angle_diff(current: float, target: float) -> float:
    """Shortest signed difference: negative = turn left, positive = turn right."""
    return (target - current + 180) % 360 - 180


# ═══════════════════════════════════════════════════════════
#  PHASE 1 — WAIT FOR GPS FIX & SAVE HOME
# ═══════════════════════════════════════════════════════════

def wait_for_fix(ser: serial.Serial) -> tuple[float, float]:
    print("\n[GPS] Waiting for valid fix  (this may take 30–60 s outdoors)…")
    dots = 0
    while True:
        data = read_gps(ser)
        if data and data['lat'] != 0.0:
            print(f"\n[GPS] Fix acquired ✓")
            print(f"      Lat: {data['lat']:.7f}  Lon: {data['lon']:.7f}")
            return data['lat'], data['lon']
        # progress dots
        print('.', end='', flush=True)
        dots += 1
        if dots % 40 == 0:
            print()


# ═══════════════════════════════════════════════════════════
#  PHASE 2 — RANDOM WANDER FOR wander_duration SECONDS
# ═══════════════════════════════════════════════════════════

MOVES = [forward, backward, turn_left, turn_right]
MOVE_NAMES = ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT']


def wander(duration: float = WANDER_DURATION):
    print(f"\n[WANDER] Random driving for {duration}s …")
    end_time = time.time() + duration
    while time.time() < end_time:
        move = random.choice(MOVES)
        secs = random.uniform(0.8, 2.0)
        remaining = end_time - time.time()
        secs = min(secs, remaining)
        print(f"  [{remaining:.1f}s left]", end=' ')
        move()
        time.sleep(secs)
    stop()
    print("[WANDER] Wander complete.\n")


# ═══════════════════════════════════════════════════════════
#  PHASE 3 — RETURN TO HOME
# ═══════════════════════════════════════════════════════════

def return_home(ser: serial.Serial, home_lat: float, home_lon: float):
    print(f"[RETURN] Navigating back to  {home_lat:.7f}, {home_lon:.7f}")
    print(f"         Stop threshold: {HOME_THRESHOLD} m\n")

    no_fix_count = 0

    while True:
        data = read_gps(ser)

        if data is None:
            no_fix_count += 1
            if no_fix_count > 10:
                print("[GPS] No fix — creeping forward to acquire signal…")
                forward()
                time.sleep(0.5)
            continue

        no_fix_count = 0
        cur_lat  = data['lat']
        cur_lon  = data['lon']
        speed    = data['speed_ms']
        cog      = data['cog']

        dist   = haversine(cur_lat, cur_lon, home_lat, home_lon)
        target = bearing_to(cur_lat, cur_lon, home_lat, home_lon)
        error  = angle_diff(cog, target)

        print(f"  dist={dist:.2f}m  speed={speed:.2f}m/s  "
              f"COG={cog:.1f}°  target={target:.1f}°  err={error:+.1f}°")

        # ── Arrived? ───────────────────────────────────────
        if dist < HOME_THRESHOLD:
            stop()
            print("\n✅  HOME REACHED!  Robot stopped.\n")
            return

        # ── Steer ──────────────────────────────────────────
        if speed < MIN_COG_SPEED:
            # Too slow to trust COG — creep forward to build speed
            print("  [slow] creeping forward…")
            forward()
        elif abs(error) <= HEADING_TOLERANCE:
            forward()
        elif error > 0:
            turn_right()
        else:
            turn_left()

        time.sleep(0.4)


# ═══════════════════════════════════════════════════════════
#  GRACEFUL SHUTDOWN
# ═══════════════════════════════════════════════════════════

def _signal_handler(sig, frame):
    print("\n[!] Interrupted — cleaning up…")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  GPS AUTO-RETURN TEST")
    print("  Pi Zero 2W  |  NEO-6M  |  L298N")
    print("=" * 55)

    ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=GPS_TIMEOUT)

    try:
        # 1️⃣  Save home position
        home_lat, home_lon = wait_for_fix(ser)
        print(f"\n[HOME] Saved  →  {home_lat:.7f}, {home_lon:.7f}")
        time.sleep(1)

        # 2️⃣  Wander randomly for 10 s
        wander(WANDER_DURATION)

        # Brief stop + pause before returning
        stop()
        time.sleep(1)

        # 3️⃣  Drive back to home
        # Drive forward briefly so GPS can establish a valid COG
        print("[RETURN] Building initial COG heading…")
        forward()
        time.sleep(2)

        return_home(ser, home_lat, home_lon)

    finally:
        ser.close()
        cleanup()
        print("[DONE] GPIO cleaned up.")


if __name__ == '__main__':
    main()
