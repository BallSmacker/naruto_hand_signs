import cv2
import mediapipe as mp
import numpy as np
import math, random, time

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
CAM_W, CAM_H  = 640, 480
SIGN_HOLD_SEC = 1.0
SIGN_TIMEOUT  = 6.0
EFFECT_DUR    = 10.0

mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils
hands_model = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
)

# ─────────────────────────────────────────
#  FINGER HELPERS
# ─────────────────────────────────────────
TIPS = [8, 12, 16, 20]
PIPS = [6, 10, 14, 18]

def fu(lms):   # fingers_up → [index,middle,ring,pinky]
    return [lms.landmark[t].y < lms.landmark[p].y for t,p in zip(TIPS,PIPS)]

def th(lms, label):  # thumb extended?
    tx,bx = lms.landmark[4].x, lms.landmark[3].x
    return tx < bx if label=="Right" else tx > bx

def fist(lms):        return not any(fu(lms))
def palm(lms):        return all(fu(lms))
def peace(lms):       f=fu(lms); return f[0] and f[1] and not f[2] and not f[3]
def idx_only(lms):    f=fu(lms); return f[0] and not f[1] and not f[2] and not f[3]
def mid_only(lms):    f=fu(lms); return not f[0] and f[1] and not f[2] and not f[3]
def ring_only(lms):   f=fu(lms); return not f[0] and not f[1] and f[2] and not f[3]
def pinky_only(lms):  f=fu(lms); return not f[0] and not f[1] and not f[2] and f[3]
def three(lms):       f=fu(lms); return f[0] and f[1] and f[2] and not f[3]
def horns(lms):       f=fu(lms); return f[0] and not f[1] and not f[2] and f[3]
def gun(lms,lab):     f=fu(lms); return f[0] and not f[1] and not f[2] and not f[3] and th(lms,lab)
def thumb_up(lms,lab):return fist(lms) and th(lms,lab)
def idx_mid_ring(lms):f=fu(lms); return f[0] and f[1] and f[2] and not f[3]  # no pinky

# ─────────────────────────────────────────
#  ANIME-ACCURATE SIGNS  (no finger overlap)
#
#  Each sign uses distinct L+R finger combos
#  readable independently by the camera.
#
#  OX:     L=peace(idx+mid)  R=peace+thumb_out
#  HARE:   L=idx only        R=peace(idx+mid)
#  MONKEY: L=palm(all 4)     R=three(idx+mid+ring)
#  DRAGON: L=fist            R=fist+thumb_out
#
#  TIGER:  L=peace           R=peace  [thumbs IN — differs from Ox by no R thumb]
#  BIRD:   L=horns(idx+pinky)R=horns(idx+pinky)
#  SNAKE:  L=palm            R=idx_only
#  RAM:    L=peace           R=idx_only
#
#  HORSE:  L=idx_only        R=palm(all 4)
#  BOAR:   L=three           R=horns
#  RAT:    L=mid_only        R=peace
# ─────────────────────────────────────────

SIGN_GUIDE = {
    # ── CHIDORI: Ox → Hare → Monkey → Dragon ──
    "Ox":     {"label":"Ox",     "how":"L: index+middle up (peace). R: index+middle up + thumb out."},
    "Hare":   {"label":"Hare",   "how":"L: index finger only. R: index+middle up (peace)."},
    "Monkey": {"label":"Monkey", "how":"L: all 4 fingers open. R: index+middle+ring up (no pinky)."},
    "Dragon": {"label":"Dragon", "how":"L: fist. R: fist + thumb out (thumbs up)."},
    # ── RASENSHURIKEN: Tiger → Bird → Snake → Ram ──
    "Tiger":  {"label":"Tiger",  "how":"L: index+middle up, thumb IN. R: index+middle up, thumb IN."},
    "Bird":   {"label":"Bird",   "how":"L: index+pinky up (horns). R: index+pinky up (horns)."},
    "Snake":  {"label":"Snake",  "how":"L: all 4 fingers open. R: index finger only pointing up."},
    "Ram":    {"label":"Ram",    "how":"L: index+middle up (peace). R: index finger only."},
    # ── FIREBALL: Horse → Boar → Rat → Tiger ──
    "Horse":  {"label":"Horse",  "how":"L: index only. R: all 4 fingers open."},
    "Boar":   {"label":"Boar",   "how":"L: index+middle+ring (three). R: index+pinky (horns)."},
    "Rat":    {"label":"Rat",    "how":"L: middle finger only. R: index+middle (peace)."},
}

def detect_sign(hands_data):
    if not hands_data: return None
    by = {lab: lms for lms,lab in hands_data}
    L = by.get("Left"); R = by.get("Right")

    # Need both hands for all signs
    if not L or not R: return None

    # ── CHIDORI signs ──
    if peace(L) and peace(R) and th(R,"Right") and not th(L,"Left"): return "Ox"
    if idx_only(L) and peace(R):                                      return "Hare"
    if palm(L) and three(R):                                          return "Monkey"
    if fist(L) and thumb_up(R,"Right"):                               return "Dragon"

    # ── RASENSHURIKEN signs ──
    # Tiger: both peace, NO thumbs (distinguishes from Ox)
    if peace(L) and peace(R) and not th(L,"Left") and not th(R,"Right"): return "Tiger"
    if horns(L) and horns(R):                                            return "Bird"
    if palm(L) and idx_only(R):                                          return "Snake"
    if peace(L) and idx_only(R):                                         return "Ram"

    # ── FIREBALL signs ──
    if idx_only(L) and palm(R):                                          return "Horse"
    if three(L) and horns(R):                                            return "Boar"
    if mid_only(L) and peace(R):                                         return "Rat"

    return None

# ─────────────────────────────────────────
#  JUTSU SEQUENCES
# ─────────────────────────────────────────
JUTSU = {
    "Chidori":       {"sequence": ["Ox",    "Hare",  "Monkey", "Dragon"]},
    "Rasenshuriken": {"sequence": ["Tiger", "Bird",  "Snake",  "Ram"]},
    "Fireball":      {"sequence": ["Horse", "Boar",  "Rat",    "Tiger"]},
}

GUIDE_LINES = [
    ("CHIDORI",       ["Ox","Hare","Monkey","Dragon"],      (255,220,80)),
    ("RASENSHURIKEN", ["Tiger","Bird","Snake","Ram"],        (255,230,60)),
    ("FIREBALL",      ["Horse","Boar","Rat","Tiger"],        (30,130,255)),
]

# ─────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────
class State:
    def __init__(self):
        self.current_sign     = None
        self.sign_start       = 0.0
        self.sequence_done    = []
        self.last_sign_time   = time.time()
        self.active_jutsu     = None
        self.jutsu_start      = 0.0
        self.particles        = []
        self.frame_count      = 0
        self.chidori_charged  = False
        self.chidori_charge_t = 0.0
        self.palm_pt          = (CAM_W // 2, CAM_H // 2)
        self.hand_r           = 70

S = State()

# ─────────────────────────────────────────
#  SEQUENCE LOGIC
# ─────────────────────────────────────────
def update_sequence(sign):
    now = time.time()
    if now - S.last_sign_time > SIGN_TIMEOUT:
        S.sequence_done = []

    if sign is None:
        S.current_sign = None; S.sign_start = 0.0; return
    if sign != S.current_sign:
        S.current_sign = sign; S.sign_start = now; return
    if now - S.sign_start >= SIGN_HOLD_SEC:
        last = S.sequence_done[-1] if S.sequence_done else None
        if sign != last:
            S.sequence_done.append(sign)
            S.last_sign_time = now
            S.sign_start     = now + 9999
            for name, data in JUTSU.items():
                seq = data["sequence"]
                if S.sequence_done[-len(seq):] == seq:
                    trigger_jutsu(name)
                    S.sequence_done = []
                    return

def trigger_jutsu(name):
    S.active_jutsu = name
    S.jutsu_start  = time.time()
    S.particles    = []

# ─────────────────────────────────────────
#  PALM / HAND SIZE HELPERS
# ─────────────────────────────────────────
def palm_center(lms):
    ids = [0, 1, 5, 9, 13, 17]
    x = sum(lms.landmark[i].x for i in ids) / len(ids)
    y = sum(lms.landmark[i].y for i in ids) / len(ids)
    return (int(x * CAM_W), int(y * CAM_H))

def hand_radius(lms):
    w = (int(lms.landmark[0].x*CAM_W), int(lms.landmark[0].y*CAM_H))
    m = (int(lms.landmark[9].x*CAM_W), int(lms.landmark[9].y*CAM_H))
    return max(30, int(math.sqrt((m[0]-w[0])**2+(m[1]-w[1])**2)*1.1))

def get_fingertips_list(lms):
    return [(int(lms.landmark[i].x*CAM_W), int(lms.landmark[i].y*CAM_H))
            for i in [4,8,12,16,20]]

# ─────────────────────────────────────────
#  DRAWING UTILITIES
# ─────────────────────────────────────────
def blend(frame, overlay, alpha):
    cv2.addWeighted(overlay, alpha, frame, 1.0-alpha, 0, frame)

def glow_circle(frame, center, radius, color, layers=6, max_alpha=0.6):
    for i in range(layers, 0, -1):
        r = int(radius * i / layers)
        if r <= 0: continue
        a  = max_alpha * (i/layers)**2
        ov = frame.copy()
        cv2.circle(ov, center, r, color, -1)
        blend(frame, ov, a)

def glow_line(frame, p1, p2, color, thickness=1, glow_w=3):
    ov = frame.copy()
    cv2.line(ov, p1, p2, color, thickness+glow_w)
    blend(frame, ov, 0.25)
    cv2.line(frame, p1, p2, color, thickness)
    cv2.line(frame, p1, p2, tuple(min(255,c+100) for c in color), max(1,thickness-1))

def jagged_bolt(frame, start, end, color, segs=8, jitter=20, thick=1):
    pts = [start]
    dx=end[0]-start[0]; dy=end[1]-start[1]
    for i in range(1, segs):
        t=i/segs
        pts.append((int(start[0]+dx*t+random.randint(-jitter,jitter)),
                    int(start[1]+dy*t+random.randint(-jitter,jitter))))
    pts.append(end)
    for i in range(len(pts)-1):
        glow_line(frame, pts[i], pts[i+1], color, thick, 3)
    if random.random()>0.55 and len(pts)>3:
        bi=random.randint(1,len(pts)-2)
        glow_line(frame, pts[bi],
                  (pts[bi][0]+random.randint(-35,35),
                   pts[bi][1]+random.randint(-35,35)), color, 1, 2)

def cast_hand_light(frame, pt, color, radius, intensity=0.18):
    ov=frame.copy(); cv2.circle(ov,pt,radius,color,-1); blend(frame,ov,intensity)

# ─────────────────────────────────────────
#  PARTICLE SYSTEM
# ─────────────────────────────────────────
def make_particle(pt, ptype, color, speed=(2,9), life=(0.4,1.0), rng=(2,8)):
    a=random.uniform(0,2*math.pi); s=random.uniform(*speed); l=random.uniform(*life)
    return {"x":float(pt[0]),"y":float(pt[1]),
            "vx":math.cos(a)*s,"vy":math.sin(a)*s,
            "life":l,"max_life":l,"r":random.randint(*rng),"color":color,"type":ptype}

def tick_particles(frame):
    alive=[]
    for p in S.particles:
        p["x"]+=p["vx"]; p["y"]+=p["vy"]
        t=p["type"]
        if t=="ember":   p["vy"]-=0.55; p["vx"]*=0.96
        elif t=="smoke": p["vy"]-=0.12; p["vx"]*=0.98; p["r"]=min(40,p["r"]+1)
        elif t=="wind":  p["vy"]+=0.04
        p["life"]-=0.022
        if p["life"]<=0: continue
        a=max(0.0,p["life"]/p["max_life"])
        c=tuple(int(ch*a) for ch in p["color"])
        cx,cy=int(p["x"]),int(p["y"])
        if t=="lightning_spark":
            cv2.line(frame,(cx,cy),(cx+int(p["vx"]*4+random.randint(-5,5)),
                                    cy+int(p["vy"]*4+random.randint(-5,5))),c,1)
        elif t=="smoke":
            ov=frame.copy(); cv2.circle(ov,(cx,cy),p["r"],c,-1); blend(frame,ov,a*0.22)
        else:
            cv2.circle(frame,(cx,cy),p["r"],c,-1)
        alive.append(p)
    S.particles=alive

# ═══════════════════════════════════════════════════════════════
#  CHIDORI
# ═══════════════════════════════════════════════════════════════
CH_WHITE=(255,255,255); CH_LBLUE=(255,230,100); CH_MBLUE=(230,160,40); CH_DBLUE=(180,90,10)

def draw_chidori_charge(frame, hands_data):
    t=time.time(); pulse=0.5+0.5*math.sin((t-S.chidori_charge_t)*16)
    for lms,_ in hands_data:
        pt=palm_center(lms); hr=hand_radius(lms)
        cast_hand_light(frame,pt,CH_MBLUE,int(hr*2.2),0.22*pulse)
        glow_circle(frame,pt,int(hr*1.6),CH_DBLUE,5,0.38*pulse)
        glow_circle(frame,pt,int(hr*1.1),CH_MBLUE,5,0.55*pulse)
        glow_circle(frame,pt,int(hr*0.6),CH_LBLUE,6,0.70*pulse)
        glow_circle(frame,pt,int(hr*0.25),CH_WHITE,4,0.90)
        for _ in range(8):
            ang=random.uniform(0,2*math.pi); dist=random.randint(int(hr*0.5),int(hr*1.8))
            jagged_bolt(frame,pt,(int(pt[0]+math.cos(ang)*dist),int(pt[1]+math.sin(ang)*dist)),
                        CH_LBLUE,7,14,1)
        for tip in get_fingertips_list(lms):
            glow_circle(frame,tip,8,CH_LBLUE,3,0.7*pulse)
            glow_circle(frame,tip,4,CH_WHITE,2,0.9)
            if random.random()>0.4:
                a2=random.uniform(0,2*math.pi)
                jagged_bolt(frame,tip,(int(tip[0]+math.cos(a2)*28),int(tip[1]+math.sin(a2)*28)),
                            CH_WHITE,4,8,1)
    if random.random()>0.3:
        S.particles.append(make_particle(S.palm_pt,"lightning_spark",CH_LBLUE,(3,12),(0.2,0.5),(1,3)))
    tick_particles(frame)

def draw_chidori_effect(frame, pt, progress):
    pulse=0.5+0.5*math.sin(time.time()*22); hr=S.hand_r; fade=max(0.0,1.0-progress*1.4)
    if progress<0.07:
        ov=frame.copy(); ov[:]=(255,240,200); blend(frame,ov,(0.07-progress)/0.07*0.92)
    cast_hand_light(frame,pt,CH_MBLUE,int(hr*3.0),0.28*fade)
    for _ in range(max(0,int(9*(1-progress*1.5)))):
        sx=random.randint(0,CAM_W)
        jagged_bolt(frame,(sx,0),(sx+random.randint(-100,100),CAM_H),CH_DBLUE,7,22,1)
    glow_circle(frame,pt,int(hr*1.9),CH_DBLUE,6,0.40*pulse*fade)
    glow_circle(frame,pt,int(hr*1.3),CH_MBLUE,6,0.55*pulse*fade)
    glow_circle(frame,pt,int(hr*0.8),CH_LBLUE,7,0.70*fade)
    glow_circle(frame,pt,int(hr*0.35),CH_WHITE,5,0.92*fade)
    for _ in range(18):
        ang=random.uniform(0,2*math.pi); dist=random.randint(int(hr*0.4),int(hr*2.0))
        jagged_bolt(frame,pt,(int(pt[0]+math.cos(ang)*dist),int(pt[1]+math.sin(ang)*dist)),
                    CH_LBLUE,random.randint(5,9),16,random.randint(1,2))
    glow_circle(frame,pt,int(hr*0.18),CH_WHITE,3,0.98*fade)
    for _ in range(4):
        S.particles.append(make_particle(pt,"lightning_spark",CH_LBLUE,(4,14),(0.15,0.4),(1,2)))
    tick_particles(frame)
    cv2.putText(frame,"CHIDORI",(CAM_W//2-105,CAM_H//2),cv2.FONT_HERSHEY_TRIPLEX,2.0,
                tuple(int(c*fade) for c in CH_WHITE),3)
    if fade>0.1:
        cv2.putText(frame,"Lightning Blade",(CAM_W//2-90,CAM_H//2+34),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,tuple(int(c*fade) for c in CH_LBLUE),1)

# ═══════════════════════════════════════════════════════════════
#  RASENSHURIKEN
# ═══════════════════════════════════════════════════════════════
RS_CYAN=(255,230,60); RS_WHITE=(255,255,220); RS_TEAL=(200,200,50)
RS_WIND=(220,210,80); RS_SKIN=(180,200,100)

def draw_rasenshuriken(frame, pt, progress):
    t=time.time(); spin=t*11+progress*6
    ease=min(1.0,progress*3.0)
    fade_out=max(0.0,1.0-(progress-0.72)/0.28) if progress>0.72 else 1.0
    eff=ease*fade_out; hr=S.hand_r
    if progress<0.09:
        ov=frame.copy(); ov[:]=(200,230,255); blend(frame,ov,(0.09-progress)/0.09*0.88)
    cast_hand_light(frame,pt,RS_SKIN,int(hr*3.5),0.22*eff)
    blade_r=int(180*ease*fade_out); blade_w=int(55*ease*fade_out)

    def draw_blade(base_angle, alpha):
        if blade_r<8 or alpha<=0: return
        sweep=1.05; steps=24; outer=[]; inner=[]
        for i in range(steps):
            frac=i/(steps-1); angle=base_angle+frac*sweep
            outer.append((int(pt[0]+math.cos(angle)*blade_r),int(pt[1]+math.sin(angle)*blade_r)))
            inner.append((int(pt[0]+math.cos(angle)*max(6,blade_r-blade_w)),
                          int(pt[1]+math.sin(angle)*max(6,blade_r-blade_w))))
        poly=np.array(outer+list(reversed(inner)),np.int32)
        ov=frame.copy(); cv2.fillPoly(ov,[poly],RS_CYAN); blend(frame,ov,alpha*0.45)
        tip_a=base_angle+sweep
        glow_line(frame,
                  (int(pt[0]+math.cos(tip_a)*max(4,blade_r-blade_w)),int(pt[1]+math.sin(tip_a)*max(4,blade_r-blade_w))),
                  (int(pt[0]+math.cos(tip_a)*blade_r),int(pt[1]+math.sin(tip_a)*blade_r)),RS_WHITE,2,4)
        for op,ip in zip(outer[::4],inner[::4]):
            ov2=frame.copy(); cv2.line(ov2,op,ip,RS_WHITE,1); blend(frame,ov2,0.18*alpha)

    for b in range(4): draw_blade(b*(math.pi/2)+spin, eff)
    if blade_r>20:
        for b in range(4): draw_blade(b*(math.pi/2)+spin-0.22, eff*0.22)
    if blade_r>8:
        for off in [0,6,14]:
            ov=frame.copy(); cv2.circle(ov,pt,blade_r+off,RS_WIND,2); blend(frame,ov,0.22*eff)
    for ring in range(7):
        phase=(progress*2.8+ring*0.16)%1.0; ring_r=int(blade_r*1.15*phase)
        ring_a=max(0.0,(1.0-phase)*0.40*eff)
        if ring_r>2:
            ov=frame.copy(); cv2.circle(ov,pt,ring_r,RS_WIND,3); blend(frame,ov,ring_a)
    core_r=max(1,int(hr*0.70*ease*fade_out))
    glow_circle(frame,pt,int(core_r*1.6),RS_TEAL,5,0.35*eff)
    glow_circle(frame,pt,core_r,RS_CYAN,7,0.80*eff)
    glow_circle(frame,pt,int(core_r*0.45),RS_WHITE,4,0.96*eff)
    for i in range(64):
        a=i*(2*math.pi/64)+spin*2.2; rs=core_r*(0.35+0.65*(i/64))
        cv2.circle(frame,(int(pt[0]+math.cos(a)*rs),int(pt[1]+math.sin(a)*rs)),2,
                   tuple(int(ch*eff) for ch in RS_CYAN),-1)
    if random.random()>0.25 and blade_r>15:
        ang=random.uniform(0,2*math.pi); dist=random.randint(blade_r//4,max(blade_r//2,blade_r))
        S.particles.append(make_particle((int(pt[0]+math.cos(ang)*dist),int(pt[1]+math.sin(ang)*dist)),
                                          "wind",RS_WHITE,(1,4),(0.3,0.7),(1,4)))
    tick_particles(frame)
    fade=max(0.0,1.0-progress*1.3)
    cv2.putText(frame,"RASENSHURIKEN",(CAM_W//2-158,CAM_H//2),cv2.FONT_HERSHEY_TRIPLEX,1.5,
                tuple(int(c*fade) for c in RS_WHITE),3)
    if fade>0.15:
        cv2.putText(frame,"Wind Release",(CAM_W//2-68,CAM_H//2+32),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,tuple(int(c*fade) for c in RS_CYAN),1)

# ═══════════════════════════════════════════════════════════════
#  FIREBALL
# ═══════════════════════════════════════════════════════════════
def fire_gradient(t):
    t=max(0.0,min(1.0,t))
    if t<0.20:   r=255;g=255;b=int(255*(1-t/0.20))
    elif t<0.45: r=255;g=int(255*(1-(t-0.20)/0.25));b=0
    elif t<0.70: r=int(255*(1-(t-0.45)/0.25*0.35));g=0;b=0
    else:        r=int(165*(1-(t-0.70)/0.30));g=0;b=0
    return (b,g,r)

def draw_fireball_effect(frame, pt, progress):
    t=time.time(); hr=S.hand_r
    grow=min(1.0,progress*2.0)
    shrink=max(0.0,1.0-(progress-0.60)/0.40) if progress>0.60 else 1.0
    eff=grow*shrink; max_r=int(hr*3.8*grow)
    if progress<0.10:
        ov=frame.copy(); ov[:]=(30,100,255); blend(frame,ov,(0.10-progress)/0.10*0.90)
    cast_hand_light(frame,pt,(30,100,255),int(max_r*1.4),0.30*eff)
    for i in range(28,0,-1):
        r=int(max_r*i/28); t_norm=1.0-(i/28)
        color=fire_gradient(t_norm*0.92); a=(i/28)**1.4*0.72*eff
        ov=frame.copy(); cv2.circle(ov,pt,r,color,-1); blend(frame,ov,a)
    core_r=max(1,int(max_r*0.20))
    glow_circle(frame,pt,core_r,(255,255,230),5,0.95*eff)
    if max_r>10:
        for i in range(28):
            ba=i*(2*math.pi/28)+t*2.5+progress
            tl=int(max_r*random.uniform(0.12,0.40)); wb=random.uniform(-0.18,0.18)
            sr=max_r-int(max_r*0.08)
            sx=int(pt[0]+math.cos(ba)*sr); sy=int(pt[1]+math.sin(ba)*sr)
            ex=int(pt[0]+math.cos(ba+wb)*(sr+tl)); ey=int(pt[1]+math.sin(ba+wb)*(sr+tl))
            col=fire_gradient(0.40+random.uniform(0,0.45))
            thick=max(1,int(5*(1-progress)*random.uniform(0.5,1.0)))
            ov=frame.copy(); cv2.line(ov,(sx,sy),(ex,ey),col,thick+2); blend(frame,ov,0.22*eff)
            cv2.line(frame,(sx,sy),(ex,ey),col,thick)
    if max_r>20:
        for _ in range(3):
            ang=random.uniform(0,2*math.pi); dr=random.randint(-6,6)
            ov=frame.copy()
            cv2.circle(ov,(int(pt[0]+math.cos(ang)*(max_r+dr)),int(pt[1]+math.sin(ang)*(max_r+dr))),
                       random.randint(4,10),(40,60,80),-1)
            blend(frame,ov,0.06*eff)
    if random.random()>0.12 and max_r>6:
        ox=pt[0]+random.randint(-max_r,max_r); oy=pt[1]+random.randint(-max_r,max_r)
        tnrm=math.sqrt((ox-pt[0])**2+(oy-pt[1])**2)/max(1,max_r)
        S.particles.append({"x":float(ox),"y":float(oy),"vx":random.uniform(-2.5,2.5),
            "vy":random.uniform(-8,-2),"life":random.uniform(0.3,0.9),"max_life":1.0,
            "r":random.randint(3,14),"color":fire_gradient(tnrm*0.65),"type":"ember"})
    if progress>0.58:
        for _ in range(3):
            sx=pt[0]+random.randint(-int(hr*1.2),int(hr*1.2))
            sy=pt[1]+random.randint(-int(hr*1.8),int(hr*0.4))
            g=random.randint(30,55)
            S.particles.append({"x":float(sx),"y":float(sy),"vx":random.uniform(-1.2,1.2),
                "vy":random.uniform(-2.5,-0.8),"life":random.uniform(0.5,1.2),"max_life":1.2,
                "r":random.randint(8,22),"color":(g,g,g),"type":"smoke"})
    tick_particles(frame)
    fade=max(0.0,1.0-progress*1.5)
    cv2.putText(frame,"KATON",(CAM_W//2-65,CAM_H//2-10),cv2.FONT_HERSHEY_TRIPLEX,1.8,
                tuple(int(c*fade) for c in fire_gradient(0.05)),3)
    if fade>0.1:
        cv2.putText(frame,"Gokakyu no Jutsu",(CAM_W//2-140,CAM_H//2+30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.85,tuple(int(c*fade) for c in fire_gradient(0.35)),2)

# ─────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────
def draw_hud(frame, sign):
    # Bottom bar
    ov=frame.copy()
    cv2.rectangle(ov,(0,CAM_H-88),(CAM_W,CAM_H),(8,8,12),-1)
    blend(frame,ov,0.88)

    # Current sign name + how-to
    if sign:
        info = SIGN_GUIDE.get(sign, {})
        label = info.get("label", sign)
        how   = info.get("how", "")
        cv2.putText(frame, f"Sign: {label}",
                    (10,CAM_H-66), cv2.FONT_HERSHEY_SIMPLEX, 0.58,(200,200,200),1)
        cv2.putText(frame, how,
                    (10,CAM_H-46), cv2.FONT_HERSHEY_SIMPLEX, 0.36,(150,150,150),1)
    else:
        cv2.putText(frame,"Show your hands and hold a sign for 1 second",
                    (10,CAM_H-56), cv2.FONT_HERSHEY_SIMPLEX, 0.40,(130,130,130),1)

    # Sequence breadcrumb
    if S.sequence_done:
        labels = [SIGN_GUIDE.get(s,{}).get("label",s) for s in S.sequence_done]
        cv2.putText(frame, "  ->  ".join(labels),
                    (10,CAM_H-22), cv2.FONT_HERSHEY_SIMPLEX, 0.52,(0,220,255),2)

    # Jutsu legend (right side)
    for i,(name,steps,col) in enumerate(GUIDE_LINES):
        cv2.putText(frame, f"{name}: {' -> '.join(steps)}",
                    (CAM_W-330, CAM_H-68+i*23), cv2.FONT_HERSHEY_SIMPLEX, 0.31, col, 1)

    # Hold progress bar
    if S.current_sign and S.sign_start < time.time():
        held=min(1.0,(time.time()-S.sign_start)/SIGN_HOLD_SEC)
        cv2.rectangle(frame,(0,CAM_H-5),(int(CAM_W*held),CAM_H),(0,255,180),-1)

    cv2.putText(frame,"R=Reset  ESC=Quit",(CAM_W-160,CAM_H-5),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,(100,100,100),1)

# ─────────────────────────────────────────
#  NEXT SIGN PROMPT  (top of frame)
# ─────────────────────────────────────────
def draw_next_sign_prompt(frame):
    """Tell the user exactly what sign to do next."""
    if S.active_jutsu or not S.sequence_done:
        return
    # Figure out which jutsu is being attempted
    for name, data in JUTSU.items():
        seq = data["sequence"]
        done = S.sequence_done
        n = len(done)
        if n < len(seq) and seq[:n] == done:
            next_sign = seq[n]
            info = SIGN_GUIDE.get(next_sign, {})
            label = info.get("label", next_sign)
            how   = info.get("how", "")
            # Highlight box
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(CAM_W,58),(20,20,30),-1)
            blend(frame, ov, 0.80)
            cv2.putText(frame, f"Next: {label}",
                        (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0,255,180), 2)
            cv2.putText(frame, how,
                        (10,46), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (180,210,180), 1)
            return

# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cv2.namedWindow("Naruto Jutsu", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Naruto Jutsu", 960, 720)
    res=None; frame_count=0

    try:
        while True:
            ok,frame=cap.read()
            if not ok: continue
            frame=cv2.flip(frame,1)
            frame_count+=1

            if frame_count%2==0:
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                res=hands_model.process(rgb)

            hands_data=[]
            if res and res.multi_hand_landmarks and res.multi_handedness:
                for lms,hd in zip(res.multi_hand_landmarks,res.multi_handedness):
                    label = hd.classification[0].label
                    mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
                    hands_data.append((lms, label))

            # Update palm tracking from first visible hand
            if hands_data:
                lms0,_=hands_data[0]
                S.palm_pt=palm_center(lms0)
                S.hand_r=hand_radius(lms0)

            sign=None
            if not S.active_jutsu:
                sign=detect_sign(hands_data)
                update_sequence(sign)

            # Active jutsu render
            if S.active_jutsu:
                elapsed=time.time()-S.jutsu_start
                progress=elapsed/EFFECT_DUR
                pt=S.palm_pt
                if progress>=1.0:
                    S.active_jutsu=None; S.particles=[]
                else:
                    if S.active_jutsu=="Chidori":        draw_chidori_effect(frame,pt,progress)
                    elif S.active_jutsu=="Rasenshuriken":draw_rasenshuriken(frame,pt,progress)
                    elif S.active_jutsu=="Fireball":     draw_fireball_effect(frame,pt,progress)

            draw_next_sign_prompt(frame)
            draw_hud(frame, sign)
            cv2.imshow("Naruto Jutsu",frame)

            key=cv2.waitKey(1)&0xFF
            if key==27: break
            elif key==ord('r'):
                S.sequence_done=[]; S.active_jutsu=None; S.particles=[]
                S.current_sign=None; S.sign_start=0.0
    finally:
        cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()