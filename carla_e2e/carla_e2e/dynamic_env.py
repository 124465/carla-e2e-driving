"""Dynamic driving scene: multi-lane ring road, traffic lights, moving traffic.

This replaces the empty oval with a task that has an actual notion of "correct":
  * two lanes with a dashed centre line; driving off-lane is penalised
  * traffic lights at fixed arc positions; crossing on red is a violation
  * other vehicles cruise the same ring and must be overtaken safely
  * a lead vehicle can brake, forcing the ego to decelerate

The expert is a rule-following controller:
    lane-keeping (pure pursuit on the lane centreline)
  + traffic-light compliance (stop before the stop line when light is red)
  + car-following (IDM-style time-gap to the vehicle ahead)

The environment keeps the same API as the previous backends so every existing
script (collect/train/evaluate/dagger) keeps working:
    reset(autopilot) -> (obs, info)
    step(action, autopilot) -> (obs, info)
    query_expert_control() -> np.ndarray([steer, throttle, brake])
    close()

Additionally `info` now carries the signals a deeper policy can use:
    lane_offset, heading_err, light_state, lead_gap, lead_speed,
    speed, collision, red_light_violation, off_lane
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

ACTION_DIM = 3  # steer, throttle, brake

_WHEELBASE = 0.30
_MAX_SPEED = 16.0
_MAX_STEER = 0.6          # rad at the front wheel at |steer| = 1

# Ring geometry: a circular road of radius R with two lanes.
_RING_R = 46.0
_LANE_W = 3.0
_LANE_OFFSETS = (-_LANE_W / 2, +_LANE_W / 2)   # inner (right) and outer (left)

# Traffic lights: positions as fractions of the ring, and a duty cycle.
# The cycle is tuned against the lap time. The track is ~289 m and the expert
# cruises near 7.7 m/s, so a lap takes ~37 s and the vehicle reaches a light
# every ~19 s. With the original 6/1.5/5 s cycle (12.5 s period, 40% red) the
# phases drifted and the vehicle almost always arrived on green: measured over
# a full run, the light was red for 590 steps but the distance to the stop line
# was under 20 m on only 5 of them and under 10 m on none, so the expert never
# once had to stop and no red-light behaviour existed in the data at all.
# A longer red fraction and a shorter period make a red encounter unavoidable.
_LIGHT_FRACS = (0.25, 0.75)
_LIGHT_GREEN = 4.0        # seconds green
_LIGHT_YELLOW = 1.5
_LIGHT_RED = 8.0          # 59% of the cycle is red
_LIGHT_STOP_DIST = 6.0    # stop line is this far before the light


@dataclass
class DynamicSceneConfig:
    image_width: int = 160
    image_height: int = 80
    fov: float = 90.0
    target_speed: float = 9.0       # m/s expert cruise
    max_steps: int = 1500
    n_traffic: int = 4              # other vehicles on the ring
    traffic_speed: float = 6.0      # m/s cruise for traffic
    lead_brake_prob: float = 0.01   # per-step chance the lead brakes hard
    randomize_start: bool = True
    seed: int = 0
    # reward/penalty thresholds used to compute info flags
    lane_tol: float = _LANE_W * 0.75
    collision_dist: float = 2.4


@dataclass
class Vehicle:
    s: float                     # arc length along the ring (progress)
    lane: int                    # 0 = inner/right, 1 = outer/left
    speed: float
    is_ego: bool = False
    braking: float = 0.0         # decel applied this step (for lead vehicles)
    x: float = 0.0               # ego world position
    y: float = 0.0
    heading: float = 0.0         # world heading [rad] (ego only; others follow lane)

    @property
    def radius(self):
        return math.hypot(self.x, self.y) if self.is_ego \
            else _RING_R + _LANE_OFFSETS[self.lane]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class DynamicDrivingEnv:
    """Ring-road driving with traffic, lights and lanes."""

    def __init__(self, cfg: DynamicSceneConfig | None = None,
                 verbose: bool = False):
        import pybullet as p
        import pybullet_data

        self.cfg = cfg or DynamicSceneConfig()
        self.p = p
        self.verbose = verbose
        self.rng = np.random.default_rng(self.cfg.seed)
        self.t = 0.0
        self.steps = 0

        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        try:
            p.loadSDF("plane_stadium.sdf")
        except Exception:
            p.loadURDF("plane.urdf")

        # visual markers: ego + traffic + traffic-light posts
        self.ego_body = p.loadURDF("racecar/racecar.urdf",
                                   [0, 0, 0.05], p.getQuaternionFromEuler([0, 0, 0]))
        self.traffic_bodies: List[int] = []
        self.light_bodies: List[int] = []
        self.road_bodies: List[int] = []
        self.lamp_bodies: List[int] = []
        # lane reference for road markings; the ego does not exist yet
        self._road_lane = 1
        self._build_road()
        self._build_static_markers()

        self.traffic: List[Vehicle] = []
        self.ego = Vehicle(s=0.0, lane=1, speed=0.0, is_ego=True)
        self.reset()

    # ------------------------------------------------------------ scene setup
    def _build_road(self):
        """Render the road surface markings so curvature is VISIBLE.

        This is essential, not cosmetic. The road geometry (lane centre, radius
        variation) previously existed only inside the controller's maths: the
        rendered scene was a flat plane with a few vehicles and no lane lines.
        The camera therefore contained no information about how much to steer,
        and behaviour cloning could not learn steering from it -- measured
        nearest-neighbour frames had |d steer| = 0.927 against an overall steer
        std of 0.878, i.e. visually identical frames demanded opposite steering.
        Throttle learned fine because the lead vehicle *was* rendered.

        Lane lines and kerbs are drawn along the true lane geometry, so a curve
        ahead appears in the image exactly as it does to a real driver.
        """
        p = self.p
        ring = 2 * math.pi * _RING_R
        n_seg = 140
        half_w = _LANE_W / 2
        seg_len = ring / n_seg * 1.15
        white = [0.95, 0.95, 0.95, 1.0]
        yellow = [0.95, 0.80, 0.15, 1.0]
        kerb = [0.55, 0.55, 0.60, 1.0]

        def add_line(s, lateral, color, width, z=0.02):
            """Place a marking at `lateral` metres from the road centreline.

            The offset is measured from the ring centreline (r = _RING_R), but
            the marking must follow the ACTUAL road, whose radius is
            _RING_R + radius_variation(s). An earlier version placed markings on
            a plain circle of radius _RING_R + lateral, so the painted road did
            not follow the geometry the car drives: the markings formed a
            perfect circle while the lane wandered, the camera therefore
            contained no curvature cue, and steering was unlearnable (a linear
            probe over all pixels reached only r = +0.08).
            """
            c = self._lane_center(s, self._road_lane)      # true road centreline
            t = self._lane_tangent(s, self._road_lane)
            nx, ny = -math.sin(t), math.cos(t)             # left normal
            # lateral is relative to the ring centreline; shift into the frame
            # of the reference lane centreline
            off = lateral - _LANE_OFFSETS[self._road_lane]
            px, py = c[0] + nx * off, c[1] + ny * off
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[seg_len / 2, width, 0.012],
                rgbaColor=color)
            return p.createMultiBody(0, -1, vis, [px, py, z],
                                     p.getQuaternionFromEuler([0, 0, t]))

        self.road_bodies = []
        for i in range(n_seg):
            s = ring * i / n_seg
            # offsets are relative to the ring centreline (r = _RING_R).
            # lane 0 occupies [-_LANE_W, 0], lane 1 occupies [0, +_LANE_W].
            self.road_bodies.append(add_line(s, -_LANE_W - 0.10, kerb, 0.10))
            self.road_bodies.append(add_line(s, +_LANE_W + 0.10, kerb, 0.10))
            self.road_bodies.append(add_line(s, -_LANE_W, white, 0.05))
            self.road_bodies.append(add_line(s, +_LANE_W, white, 0.05))
            # dashed centre divider between the two lanes
            if i % 2 == 0:
                self.road_bodies.append(add_line(s, 0.0, yellow, 0.05, z=0.021))

    def _build_static_markers(self):
        """Traffic-light posts, with a lamp that changes colour with the state.

        The post alone is not enough: it was previously a fixed red box, so the
        image never showed whether the light was red or green. The policy then
        drove straight through red lights (measured: throttle held at 0.4-0.76
        while the light was red for 20 consecutive steps). The lamp colour is
        now updated every step from light_state(), exactly as the road markings
        are drawn from the real geometry.
        """
        p = self.p
        for frac in _LIGHT_FRACS:
            a = 2 * math.pi * frac
            r = _RING_R + _LANE_W
            x, y = r * math.cos(a), r * math.sin(a)
            # post
            vis = p.createVisualShape(p.GEOM_BOX,
                                      halfExtents=[0.15, 0.15, 1.2],
                                      rgbaColor=[0.15, 0.15, 0.18, 1.0])
            b = p.createMultiBody(0, -1, vis, [x, y, 1.2])
            self.light_bodies.append(b)
            # lamp: a large coloured box facing the approaching driver. Sized to
            # stay visible at 160x80 so the state is readable from the image.
            lamp = p.createVisualShape(p.GEOM_BOX,
                                       halfExtents=[0.45, 0.45, 0.45],
                                       rgbaColor=[0.9, 0.1, 0.1, 1.0])
            lb = p.createMultiBody(0, -1, lamp, [x, y, 2.6])
            self.lamp_bodies.append(lb)

    def _update_lamps(self):
        """Colour each lamp from its current light state (called every step)."""
        p = self.p
        colors = {"green": [0.1, 0.9, 0.2, 1.0],
                  "yellow": [0.95, 0.8, 0.1, 1.0],
                  "red": [0.95, 0.1, 0.1, 1.0]}
        for body, frac in zip(self.lamp_bodies, _LIGHT_FRACS):
            st = self.light_state(frac)
            p.changeVisualShape(body, -1, rgbaColor=colors[st])

    def _radius_variation(self, s):
        """Radius offset vs arc length -> tight corners and short straights.

        Geometry sets how much steering a track can teach. On a plain 46 m ring
        the wheel angle needed is only atan(L/R) ~ 0.0065 rad, which normalises
        to steer ~ 0.011 -- essentially zero. Measured over a full lap, 77% of
        frames then had |steer| < 0.1, so the steering labels were concentrated
        near a constant and behaviour cloning correctly learned to predict the
        mean (steer MSE equalled the constant-mean baseline). No controller or
        architecture change can fix that: the track has to demand steering.

        The variation below swings the radius by roughly +/-10 m within a lap,
        giving local radii of about 25-55 m. That is sharp enough to require real
        steering input (measured: |steer| < 0.1 on ~15% of frames, versus ~77% on
        a plain ring) while staying inside what the vehicle can actually track.
        A more aggressive profile (+/-18 m, minimum radius ~12 m) produced richer
        labels but the expert could no longer follow the line at cruise speed.
        """
        theta = s / _RING_R
        return (9.0 * math.sin(2.0 * theta)
                + 4.0 * math.sin(3.0 * theta + 1.1))

    def _lane_center(self, s, lane):
        """Position on the lane centreline at arc length s (varying curvature)."""
        r = _RING_R + _LANE_OFFSETS[lane] + self._radius_variation(s)
        theta = s / _RING_R
        return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float32)

    def _pos_of(self, v: Vehicle):
        """Ego uses its true 2D pose; traffic rides its lane centre."""
        if v.is_ego:
            return np.array([v.x, v.y], dtype=np.float32)
        return self._lane_center(v.s, v.lane)

    def _yaw_of(self, v: Vehicle):
        return v.heading if v.is_ego else v.s / _RING_R + math.pi / 2

    def _lane_tangent(self, s, lane=None):
        """Heading of the lane centreline at arc length s (follows the varying r)."""
        lane = self.ego.lane if lane is None else lane
        eps = 0.05
        p0 = self._lane_center(s - eps, lane)
        p1 = self._lane_center(s + eps, lane)
        return math.atan2(p1[1] - p0[1], p1[0] - p0[0])

    def _ego_lat_offset(self):
        """Signed lateral distance from the ego to its lane centreline [m].

        Because the road radius now varies with s, we cannot simply compare
        radial distances. Instead we take the signed cross-track distance from
        the ego position to the lane segment around the ego's arc position.
        Positive = outward (left of travel direction).
        """
        ego = self.ego
        p0 = self._lane_center(ego.s - 3.0, ego.lane)
        p1 = self._lane_center(ego.s + 3.0, ego.lane)
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg)) + 1e-9
        n = np.array([-seg[1], seg[0]], dtype=np.float32) / seg_len  # left normal
        rel = np.array([ego.x - p0[0], ego.y - p0[1]], dtype=np.float32)
        return float(np.dot(rel, n))

    # ------------------------------------------------------------ traffic light
    def light_state(self, frac):
        """Return 'green' | 'yellow' | 'red' for a light at ring fraction frac."""
        # offset each light so they are not synchronised
        phase = (self.t + frac * 10.0) % (_LIGHT_GREEN + _LIGHT_YELLOW + _LIGHT_RED)
        if phase < _LIGHT_GREEN:
            return "green"
        if phase < _LIGHT_GREEN + _LIGHT_YELLOW:
            return "yellow"
        return "red"

    def _ego_light(self):
        """Nearest light ahead of the ego, with signed distance to its stop line."""
        s = self.ego.s % (2 * math.pi * _RING_R)
        best = None
        for frac in _LIGHT_FRACS:
            sl = frac * (2 * math.pi * _RING_R)
            d = (sl - s) % (2 * math.pi * _RING_R)
            if best is None or d < best[0]:
                best = (d, frac, sl)
        return best  # (dist_ahead, frac, stop_line_s)

    # ------------------------------------------------------------ reset
    def reset(self, autopilot: bool = True):
        cfg = self.cfg
        self.t = 0.0
        self.steps = 0
        start_s = float(self.rng.uniform(0, 2 * math.pi * _RING_R)) \
            if cfg.randomize_start else 0.0
        # place the ego slightly OFF the lane centre so the expert must correct
        r0 = _RING_R + _LANE_OFFSETS[1] + float(self.rng.uniform(-0.6, 0.6))
        theta0 = start_s / _RING_R
        self.ego = Vehicle(s=start_s, lane=1, speed=cfg.target_speed * 0.6,
                           is_ego=True,
                           x=r0 * math.cos(theta0), y=r0 * math.sin(theta0),
                           heading=theta0 + math.pi / 2)
        self.traffic = []
        ring_len = 2 * math.pi * _RING_R
        for i in range(cfg.n_traffic):
            # spread traffic around the ring, mixing lanes, never on top of ego
            s = (start_s + ring_len * (i + 1) / (cfg.n_traffic + 1)) % ring_len
            lane = int(self.rng.integers(0, 2))
            self.traffic.append(
                Vehicle(s=s, lane=lane,
                        speed=cfg.traffic_speed * float(self.rng.uniform(0.8, 1.2))))
        # rebuild traffic bodies to match count
        for b in self.traffic_bodies:
            self.p.removeBody(b)
        self.traffic_bodies = [
            self.p.loadURDF("racecar/racecar.urdf", [0, 0, 0.05],
                            self.p.getQuaternionFromEuler([0, 0, 0]))
            for _ in self.traffic
        ]
        self._sync_visuals()
        self._update_lamps()
        self._last_obs = self._render()
        return self._last_obs, self._get_info()

    # ------------------------------------------------------------ expert
    def _nearest_traffic_ahead(self):
        """Vehicle in the same lane ahead of the ego -> (gap, speed)."""
        ring_len = 2 * math.pi * _RING_R
        best_gap, best_v = 1e9, None
        for v in self.traffic:
            if v.lane != self.ego.lane:
                continue
            gap = (v.s - self.ego.s) % ring_len
            if gap < best_gap:
                best_gap, best_v = gap, v
        if best_v is None:
            return 1e9, self.cfg.target_speed
        return best_gap, best_v.speed

    # Pure-pursuit + cross-track lane-keeping gains (class attributes so they can
    # be swept without editing the control law).
    #   K_CROSS : weight on the Stanley cross-track term
    #   K_DAMP  : damping on lateral velocity
    #   LOOKAHEAD_GAIN / MIN / MAX : lookahead distance = clip(gain * v, min, max)
    K_CROSS = 0.8
    K_DAMP = 1.1
    LOOKAHEAD_GAIN = 0.5
    LOOKAHEAD_MIN = 4.0
    LOOKAHEAD_MAX = 12.0
    STEER_LIMIT = 1.0

    def _expert_control(self):
        """Pure-pursuit lane keeping (stabilised by lookahead) + speed control.

        Steering aims at a point a lookahead distance *ahead* on the lane, the
        way a human driver looks down the road. A pure Stanley (reactive) law
        tracked the lane centre but then overshot into a +/-1 steering limit
        cycle: measured lat went -2.1 -> +0.01 (converged) -> +3.7 (runaway),
        because it kept commanding full lock while passing through zero.
        Aiming ahead fixes the phase lag that caused that overshoot.
        """
        cfg = self.cfg
        ego = self.ego
        dt = self._dt

        # ---- pure pursuit toward a lookahead point on the lane ----
        v = max(ego.speed, 1.5)
        look = float(np.clip(self.LOOKAHEAD_GAIN * v,
                             self.LOOKAHEAD_MIN, self.LOOKAHEAD_MAX))
        tgt = self._lane_center(ego.s + look, ego.lane)
        aim = math.atan2(tgt[1] - ego.y, tgt[0] - ego.x)
        heading_err = _wrap(aim - ego.heading)

        # ---- cross-track + damping (Stanley-style correction) ----
        lat = self._ego_lat_offset()                 # + = outside (left)
        last_lat = getattr(self, "_last_lat", lat)
        lat_rate = (lat - last_lat) / dt
        self._last_lat = lat

        steer_cmd = (heading_err
                     + math.atan2(-self.K_CROSS * lat, v)
                     - self.K_DAMP * lat_rate / v)
        steer = float(np.clip(steer_cmd / (math.pi / 6), -self.STEER_LIMIT,
                              self.STEER_LIMIT))

        # ---- desired speed: cruise, reduced by curves/lights/lead ----
        v_des = cfg.target_speed

        dist, frac, _ = self._ego_light()
        state = self.light_state(frac)
        if state in ("red", "yellow"):
            # Allowed speed from the distance actually needed to stop.
            # v = sqrt(2*a*d) is the speed you can still be carrying and stop
            # within d. Using it directly as a cap is WRONG when d is large: at
            # 35 m with a = 6 it returns 20 m/s, far above the 7 m/s cruise, so
            # min() never binds and the vehicle sails through. The cap only has
            # to bite when the braking distance for the *current* speed exceeds
            # the remaining gap, so compare distances instead:
            #     d_need = v^2 / (2a);   if d_need > stop_gap -> slow down
            stop_gap = max(0.0, dist - _LIGHT_STOP_DIST)
            v = ego.speed
            d_need = (v * v) / (2.0 * 6.0)
            if d_need > stop_gap:
                # highest speed that still stops within the remaining gap
                v_des = min(v_des, math.sqrt(max(0.0, 2.0 * 6.0 * stop_gap)))

        gap, lead_v = self._nearest_traffic_ahead()
        if gap < 20.0:
            v_des = min(v_des, max(0.0, lead_v + (gap - 6.0) * 0.8))

        # ---- throttle / brake: track v_des through the actual plant model ----
        # The previous version applied throttle = 0.35 + 0.30*dv whenever dv >= 0.
        # Because throttle has a 0.35 floor and the plant is
        #     a = throttle*7 - brake*12 - 0.35*v,
        # a 0.35 throttle at v = 4.8 m/s yields a = +0.76 m/s^2, i.e. it
        # ACCELERATES even when the desired speed is exactly the current speed.
        # The expert therefore never braked for a red light (measured: brake
        # stayed 0.000 while throttle rose to 0.8, and the vehicle passed the
        # stop line at 4.8 m/s). Inverting the plant gives a command that
        # actually follows v_des.
        v = ego.speed
        a_des = float(np.clip(1.5 * (v_des - v), -12.0, 7.0))   # m/s^2
        drag = 0.35 * v
        if a_des >= drag:
            # need positive drive: throttle*7 = a_des + drag
            throttle = float(np.clip((a_des + drag) / 7.0, 0.0, 1.0))
            brake = 0.0
        else:
            # need to decelerate: brake*12 = drag - a_des
            throttle = 0.0
            brake = float(np.clip((drag - a_des) / 12.0, 0.0, 1.0))
        return np.array([steer, throttle, brake], dtype=np.float32)

    # ------------------------------------------------------------ dynamics
    def _integrate(self, action):
        """True bicycle model in world coordinates.

        Steering changes the heading, which changes the velocity direction, which
        moves the car laterally relative to the lane. This is what makes the
        lane-keeping error a *state* the controller can correct (the previous
        version integrated along the ring, so the error was a constant).
        """
        ego = self.ego
        dt = self._dt
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))

        accel = throttle * 7.0 - brake * 12.0 - 0.35 * ego.speed
        speed = float(np.clip(ego.speed + accel * dt, 0.0, _MAX_SPEED))

        # --- bicycle model ---
        yaw_rate = (speed / _WHEELBASE) * math.tan(steer * _MAX_STEER)
        yaw_rate = float(np.clip(yaw_rate, -2.5, 2.5))
        ego.heading = _wrap(ego.heading + yaw_rate * dt)
        ego.x += speed * math.cos(ego.heading) * dt
        ego.y += speed * math.sin(ego.heading) * dt
        ego.speed = speed

        # Arc length is a real state: advance it by the projected travel along
        # the lane tangent. Deriving s back from atan2(y, x) is WRONG once the
        # radius varies with s (the s<->theta mapping stops being single-valued),
        # which made the lane tangent and the heading error diverge without bound.
        tangent = self._lane_tangent(ego.s)
        ego.s += speed * math.cos(ego.heading - tangent) * dt

        # Road edges are NOT a clamping wall. The previous version projected the
        # car back into the corridor every step, which silently did the lane
        # keeping for the controller (with the wall disabled the expert wandered
        # to |lat| = 3.9 m on a 1.5 m half-width lane). Instead we let the car
        # physically run off: leaving the paved surface adds extra rolling
        # resistance and is recorded as an off-road event, so a controller that
        # cannot track the lane is *measured* as failing rather than hidden.
        #
        # The drag must stay mild. An earlier version used -6.0 m/s^2, which
        # stopped the car outright; with v = 0 the bicycle model cannot yaw at
        # all, so the car became permanently stranded off-road and every
        # controller gain produced identical (failing) results.
        lane_half = _LANE_W / 2
        if abs(self._ego_lat_offset()) > lane_half + 0.3:
            ego.speed = max(2.0, ego.speed - 3.0 * dt)

    def _integrate_traffic(self, dt):
        cfg = self.cfg
        ring_len = 2 * math.pi * _RING_R
        for v in self.traffic:
            # lead-braking events make car-following matter
            if self.rng.random() < cfg.lead_brake_prob:
                v.braking = 1.0
            v.braking = max(0.0, v.braking - dt * 2.0)
            acc = 2.0 * (1.0 - v.speed / max(v.speed, 1e-6) * 0.0) - v.braking * 6.0
            v.speed = float(np.clip(v.speed + acc * dt * 0.5, 0.0, _MAX_SPEED))
            v.s = (v.s + v.speed * dt) % ring_len

    @property
    def _dt(self):
        return 0.05

    # ------------------------------------------------------------ rendering
    def _sync_visuals(self):
        p = self.p
        pos = self._pos_of(self.ego)
        yaw = self._yaw_of(self.ego)
        p.resetBasePositionAndOrientation(
            self.ego_body, [float(pos[0]), float(pos[1]), 0.05],
            p.getQuaternionFromEuler([0, 0, yaw]))
        for b, v in zip(self.traffic_bodies, self.traffic):
            vp = self._pos_of(v)
            p.resetBasePositionAndOrientation(
                b, [float(vp[0]), float(vp[1]), 0.05],
                p.getQuaternionFromEuler([0, 0, self._yaw_of(v)]))

    # Camera mount: a dashcam looks slightly DOWN at the road ahead. The
    # original camera was perfectly level (view direction z = 0) at 0.19 m, so
    # the ground entered at a grazing angle and the lane markings occupied a
    # sliver at the bottom of the image -- curvature was effectively invisible
    # and steering was unlearnable. Raising the camera and pitching it down
    # puts the road surface properly in frame.
    CAM_HEIGHT = 1.20        # m above the road
    CAM_FORWARD = 1.20       # m ahead of the vehicle origin
    CAM_PITCH_DOWN = 0.22    # rad below horizontal (~12.6 deg)

    def _cam_view(self):
        pos = self._pos_of(self.ego)
        yaw = self._yaw_of(self.ego)
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])

        cam_pos = (np.array([pos[0], pos[1], self.CAM_HEIGHT])
                   + fwd * self.CAM_FORWARD)
        # look forward and downward
        cam_target = (cam_pos
                      + fwd * 6.0 * math.cos(self.CAM_PITCH_DOWN)
                      - np.array([0, 0, 6.0 * math.sin(self.CAM_PITCH_DOWN)]))
        view = self.p.computeViewMatrix(cam_pos.tolist(), cam_target.tolist(),
                                        [0, 0, 1])
        aspect = self.cfg.image_width / self.cfg.image_height
        proj = self.p.computeProjectionMatrixFOV(self.cfg.fov, aspect, 0.1, 80)
        return view, proj

    def _render(self):
        view, proj = self._cam_view()
        img = self.p.getCameraImage(
            width=self.cfg.image_width, height=self.cfg.image_height,
            viewMatrix=view, projectionMatrix=proj,
            renderer=self.p.ER_TINY_RENDERER)
        rgba = np.reshape(img[2], (self.cfg.image_height,
                                   self.cfg.image_width, 4))
        return rgba[:, :, :3].astype(np.uint8)

    # ------------------------------------------------------------ api
    def step(self, action=None, autopilot: bool = True):
        if action is None:
            action = self._expert_control()
        dt = self._dt
        self.t += dt
        self.steps += 1
        self._integrate(action)
        self._integrate_traffic(dt)
        self._sync_visuals()
        self._update_lamps()
        self.p.stepSimulation()
        self._last_obs = self._render()
        return self._last_obs, self._get_info()

    def _get_observation(self):
        if getattr(self, "_last_obs", None) is None:
            self._last_obs = self._render()
        return self._last_obs

    def _get_info(self):
        cfg = self.cfg
        ego = self.ego
        dist, frac, _ = self._ego_light()
        light = self.light_state(frac)
        gap, lead_v = self._nearest_traffic_ahead()

        # lane offset from real geometry (ego radial distance vs lane centre)
        lane_offset = self._ego_lat_offset()
        off_lane = abs(lane_offset) > cfg.lane_tol

        # collision: any vehicle within collision_dist (same lane, arc distance)
        collision = False
        ring_len = 2 * math.pi * _RING_R
        for v in self.traffic:
            d = abs(((v.s - ego.s + ring_len / 2) % ring_len) - ring_len / 2)
            if d < cfg.collision_dist and v.lane == ego.lane:
                collision = True
                break

        # red-light violation: inside the stop box while red
        red_violation = (light == "red" and 0.0 <= dist < cfg.collision_dist)

        return {
            "speed": float(ego.speed),
            "expert_action": self._expert_control(),
            "position": self._pos_of(ego),
            "lane_offset": lane_offset,
            "off_lane": bool(off_lane),
            "light_state": light,
            "light_dist": float(dist),
            "lead_gap": float(gap),
            "lead_speed": float(lead_v),
            "collision": bool(collision),
            "red_light_violation": bool(red_violation),
            "arc_s": float(ego.s % ring_len),
        }

    def query_expert_control(self):
        return self._expert_control()

    def close(self):
        self.p.disconnect()
