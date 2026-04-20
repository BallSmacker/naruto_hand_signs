import cv2
import mediapipe as mp
import numpy as np
import random
import math

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
CAM_W, CAM_H = 640, 480
DRAW_COLOR   = (0, 255, 120)
ERASE_RADIUS = 40
SMOOTH_ALPHA = 0.4
MIN_MOVE     = 2

COLORS = [
    ("Green",  (0, 255, 120)),
    ("Blue",   (255, 120, 0)),
    ("Red",    (0, 80, 255)),
    ("Yellow", (0, 220, 255)),
    ("White",  (255, 255, 255)),
]
COLOR_BOX_SIZE = 40


# ─────────────────────────────────────────
#  MEDIAPIPE INIT
# ─────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_draw    = mp.solutions.drawing_utils
hands_model = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)


# ─────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────
class AppState:
    def __init__(self):
        self.mode        = "idle"     # idle | draw | erase | effect
        self.draw_color  = COLORS[0][1]
        self.canvas      = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        self.prev_pt     = None
        self.smooth_pt   = None
        self.particles   = []         # list of Particle dicts

state = AppState()


# ─────────────────────────────────────────
#  GESTURE DETECTION
# ─────────────────────────────────────────
TIPS = [8, 12, 16, 20]   # index, middle, ring, pinky tip IDs
PIPS = [6, 10, 14, 18]   # corresponding second-joint IDs

def fingers_up(lms, w, h):
    """Return list of booleans [index, middle, ring, pinky]."""
    pts = [(int(lms.landmark[i].x * w), int(lms.landmark[i].y * h)) for i in range(21)]
    up = []
    for tip, pip in zip(TIPS, PIPS):
        up.append(pts[tip][1] < pts[pip][1])   # tip above pip → finger up
    return up

def thumb_up(lms, w, h):
    """Rough check: thumb tip is clearly to the left of thumb base (mirrored cam)."""
    tx = lms.landmark[4].x
    bx = lms.landmark[2].x
    return tx < bx - 0.04

def detect_gesture(lms, w, h):
    up   = fingers_up(lms, w, h)
    n    = sum(up)
    fist = n == 0 and not thumb_up(lms, w, h)

    if fist:            return "clear"
    if n == 1 and up[0]: return "draw"
    if n == 2 and up[0] and up[1]: return "erase"
    if n == 3:          return "effect"
    return "idle"


# ─────────────────────────────────────────
#  SMOOTHING
# ─────────────────────────────────────────
def smooth(new_pt):
    if state.smooth_pt is None:
        state.smooth_pt = new_pt
        return new_pt
    sx = int(SMOOTH_ALPHA * new_pt[0] + (1 - SMOOTH_ALPHA) * state.smooth_pt[0])
    sy = int(SMOOTH_ALPHA * new_pt[1] + (1 - SMOOTH_ALPHA) * state.smooth_pt[1])
    state.smooth_pt = (sx, sy)
    return (sx, sy)


# ─────────────────────────────────────────
#  PARTICLES  (effect mode)
# ─────────────────────────────────────────
def spawn_particles(pt, n=6):
    for _ in range(n):
        angle  = random.uniform(0, 2 * math.pi)
        speed  = random.uniform(2, 7)
        color  = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(0, 100),
        )
        state.particles.append({
            "x": float(pt[0]), "y": float(pt[1]),
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
            "life": 1.0,
            "radius": random.randint(4, 9),
            "color": color,
        })

def update_draw_particles(frame):
    alive = []
    for p in state.particles:
        p["x"]    += p["vx"]
        p["y"]    += p["vy"]
        p["vy"]   += 0.3            # gravity
        p["life"] -= 0.05
        if p["life"] <= 0:
            continue
        alpha  = p["life"]
        color  = tuple(int(c * alpha) for c in p["color"])
        center = (int(p["x"]), int(p["y"]))
        cv2.circle(frame, center, p["radius"], color, -1)
        alive.append(p)
    state.particles = alive


# ─────────────────────────────────────────
#  DRAWING ACTIONS
# ─────────────────────────────────────────
def handle_draw(pt):
    if state.prev_pt is None:
        state.prev_pt = pt
        return
    dx = abs(pt[0] - state.prev_pt[0])
    dy = abs(pt[1] - state.prev_pt[1])
    if dx + dy > MIN_MOVE:
        cv2.line(state.canvas, state.prev_pt, pt, state.draw_color, 8, cv2.LINE_AA )
        state.prev_pt = pt

def handle_erase(pt):
    cv2.circle(state.canvas, pt, ERASE_RADIUS, (0, 0, 0), -1)
    state.prev_pt = None

def handle_effect(pt, frame):
    spawn_particles(pt)
    cv2.circle(frame, pt, 18, (0, 200, 255), 2)
    state.prev_pt = None

def reset_stroke():
    state.prev_pt = None


# ─────────────────────────────────────────
#  COLOR SELECTOR UI
# ─────────────────────────────────────────
COLOR_BOX_Y = 10

def draw_color_selector(frame):
    for i, (name, color) in enumerate(COLORS):
        x1 = 10 + i * (COLOR_BOX_SIZE + 8)
        y1 = COLOR_BOX_Y
        x2, y2 = x1 + COLOR_BOX_SIZE, y1 + COLOR_BOX_SIZE
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        if color == state.draw_color:
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)

def check_color_tap(pt):
    """If fingertip is inside a color box, switch color."""
    x, y = pt
    for i, (name, color) in enumerate(COLORS):
        x1 = 10 + i * (COLOR_BOX_SIZE + 8)
        y1 = COLOR_BOX_Y
        x2, y2 = x1 + COLOR_BOX_SIZE, y1 + COLOR_BOX_SIZE
        if x1 <= x <= x2 and y1 <= y <= y2:
            state.draw_color = color
            return True
    return False


# ─────────────────────────────────────────
#  HUD / OVERLAY
# ─────────────────────────────────────────
MODE_COLORS = {
    "draw":   (0, 255, 120),
    "erase":  (0, 140, 255),
    "effect": (200, 0, 255),
    "idle":   (160, 160, 160),
    "clear":  (0, 50, 255),
}
GESTURE_HINTS = {
    "draw":   "✏  DRAW  [1 finger]",
    "erase":  "◌  ERASE  [2 fingers]",
    "effect": "✦  EFFECT  [3 fingers]",
    "idle":   "—  IDLE",
    "clear":  "✕  CLEAR  [fist]",
}

def draw_hud(frame, mode):
    color = MODE_COLORS.get(mode, (200, 200, 200))
    label = GESTURE_HINTS.get(mode, mode)
    cv2.rectangle(frame, (0, CAM_H - 48), (380, CAM_H), (20, 20, 20), -1)
    cv2.putText(frame, label, (12, CAM_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    # Gesture guide (top-right)
    hints = [
        "1 finger = Draw",
        "2 fingers = Erase",
        "3 fingers = Effect",
        "Fist = Clear",
        "C = Clear  ESC = Quit",
    ]
    for i, h in enumerate(hints):
        cv2.putText(frame, h, (CAM_W - 260, 30 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)


# ─────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    cv2.namedWindow("Hand Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Tracker", 1200, 720)

    try:
        frame_count = 0
        while True:
            ok, frame = cap.read()
            frame_count += 1
            if frame_count % 2 == 0:
                res = hands_model.process(rgb)

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands_model.process(rgb)

            gesture = "idle"

            if res.multi_hand_landmarks:
                lms = res.multi_hand_landmarks[0]   # first hand only
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

                # Fingertip position (index finger tip = landmark 8)
                raw_pt = (
                    int(lms.landmark[8].x * CAM_W),
                    int(lms.landmark[8].y * CAM_H),
                )
                pt      = smooth(raw_pt)
                gesture = detect_gesture(lms, CAM_W, CAM_H)

                # Color tap check (draw mode only)
                if gesture == "draw":
                    check_color_tap(pt)

                # Mode-based actions
                if gesture == "draw":
                    state.mode = "draw"
                    handle_draw(pt)
                elif gesture == "erase":
                    state.mode = "erase"
                    handle_erase(pt)
                elif gesture == "effect":
                    state.mode = "effect"
                    handle_effect(pt, frame)
                elif gesture == "clear":
                    state.mode = "idle"
                    state.canvas[:] = 0
                    reset_stroke()
                else:
                    state.mode = "idle"
                    reset_stroke()

                # Fingertip dot
                dot_color = MODE_COLORS.get(state.mode, (200, 200, 200))
                cv2.circle(frame, pt, 10, dot_color, -1)

            else:
                state.mode = "idle"
                state.smooth_pt = None
                reset_stroke()

            # Compose: camera + canvas overlay
            combined = cv2.addWeighted(frame, 0.55, state.canvas, 0.85, 0)

            # Particles drawn on top
            update_draw_particles(combined)

            # UI
            draw_color_selector(combined)
            draw_hud(combined, state.mode)

            cv2.imshow("Hand Tracker", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:             # ESC → quit
                break
            elif key == ord('c'):     # C → clear canvas
                state.canvas[:] = 0

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()