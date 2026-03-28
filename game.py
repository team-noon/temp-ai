"""
soccer_env.py
=============
A Gymnasium-compatible wrapper around a Pygame 2v2 soccer simulation.

Install dependencies:
    pip install gymnasium stable-baselines3 pygame numpy

Quick start:
    from soccer_env import SoccerEnv
    env = SoccerEnv(render_mode="human")

    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

Training with SB3:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    env = SoccerEnv(render_mode=None)   # headless for training
    check_env(env)                      # sanity-check the wrapper

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500_000)
    model.save("soccer_ppo")
"""

import math
import random
from typing import Optional

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
WIDTH        = 900
HEIGHT       = 600
GOAL_HEIGHT  = 260
FPS          = 60
MAX_STEPS    = 2_000       # episode time limit

# Colours
GREEN  = (34, 139, 34)
WHITE  = (255, 255, 255)
RED    = (220, 50,  50)
BLUE   = (50,  100, 220)
YELLOW = (255, 215, 0)
BLACK  = (0,   0,   0)


class Vec2:
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y
    def __add__(self, other: Vec2):
        return Vec2(self.x + other.x, self.y + other.y)
    def __sub__(self, other: Vec2):
        return Vec2(self.x - other.x, self.y - other.y)
    def __mul__(self, num):
        return Vec2(self.x * num, self.y * num)
    def __truediv__(self, num):
        return Vec2(self.x / num, self.y / num)
    def __iadd__(self, other: Vec2):
        self.x += other.x
        self.y += other.y
        return self
    def __isub__(self, other: Vec2):
        self.x -= other.x
        self.y -= other.y
        return self
    def __imul__(self, num):
        self.x *= num
        self.y *= num
        return self
    def __itruediv__(self, num):
        self.x /= num
        self.y /= num
        return self
    def vectorDist(self, other: Vec2):
        return Vec2(other.x - self.x, other.y - self.y)
    def dist(self, other: Vec2):
        vecDist = self.vectorDist(other)
        return np.sqrt(vecDist.x*vecDist.x + vecDist.y*vecDist.y)
    def vectorDistNormalized(self, other: Vec2):
        vecDist = self.vectorDist(other)
        hypo = np.sqrt(vecDist.x*vecDist.x + vecDist.y*vecDist.y)
        if (hypo <= 0):
            return Vec2(0, 0)
        return vecDist.__truediv__(hypo)


# ──────────────────────────────────────────────
# Physics objects
# ──────────────────────────────────────────────
class _Ball:
    MAX_SPEED = 20.0
    RADIUS   = 10
    FRICTION = 0.98

    def __init__(self):
        self.coordinate : Vec2 = Vec2(0, 0)
        self.speed : Vec2 = Vec2(0, 0)
        self.reset()

    def reset(self, cx=WIDTH // 2, cy=HEIGHT // 2):
        self.coordinate.x = cx
        self.coordinate.y = cy
        self.speed.x = random.uniform(-3, 3)
        self.speed.y = random.uniform(-2, 2)

    def step(self):
        self.coordinate += self.speed
        self.speed *= self.FRICTION
        spd = math.hypot(self.speed.x, self.speed.y)
        if spd > self.MAX_SPEED:
            self.speed = self.speed * (self.MAX_SPEED / spd)
        gt = (HEIGHT - GOAL_HEIGHT) // 2
        gb = gt + GOAL_HEIGHT


        # Top / bottom walls
        if (not (self.RADIUS < self.coordinate.y < HEIGHT - self.RADIUS)):
            self.coordinate.y = max(self.RADIUS, min(HEIGHT - self.RADIUS, self.coordinate.y))
            self.speed.y *= -1
        if (not (self.RADIUS < self.coordinate.x < WIDTH - self.RADIUS) and not (gt < self.coordinate.y < gb)):
            self.coordinate.x = max(self.RADIUS, min(WIDTH - self.RADIUS, self.coordinate.x))
            self.speed.x *= -1

    def kick(self, displacement: Vec2, power=12.0):
        dp = displacement + self.coordinate
        self.speed = self.coordinate.vectorDistNormalized(dp) * power

    def draw(self, surf):
        p = (int(self.coordinate.x), int(self.coordinate.y))
        pygame.draw.circle(surf, YELLOW, p, self.RADIUS)
        pygame.draw.circle(surf, BLACK,  p, self.RADIUS, 2)


class _Player:
    """Kinematic player driven by (dx, dy, kick) actions."""
    RADIUS     = 16
    KICK_RANGE = 28
    KICK_COOLDOWN_FRAMES = FPS * 1

    def __init__(self, _x, _y, color):
        self.coordinate : Vec2 = Vec2(_x, _y)
        self.color      = color
        self._kick_cooldown = 0
    
    def tick_cooldown(self):
        if self._kick_cooldown > 0:
            self._kick_cooldown -= 1

    def move(self, displacement: Vec2, speed = 4.0):
        dp = displacement + self.coordinate
        self.coordinate += self.coordinate.vectorDistNormalized(dp) * speed
        self.coordinate.x  = np.clip(self.coordinate.x, self.RADIUS, WIDTH  - self.RADIUS)
        self.coordinate.y  = np.clip(self.coordinate.y, self.RADIUS, HEIGHT - self.RADIUS)

    def reset(self, _x, _y):
        self.coordinate.x = _x
        self.coordinate.y = _y

    def draw(self, surf):
        p = (int(self.coordinate.x), int(self.coordinate.y))
        pygame.draw.circle(surf, self.color, p, self.RADIUS)
        pygame.draw.circle(surf, BLACK,      p, self.RADIUS, 2)
    
    @property
    def can_kick(self) -> bool:
        return self._kick_cooldown == 0

    def try_kick(self, ball: _Ball, displacement: Vec2, power=12.0):
        if not self.can_kick:
            return False                 # still on cooldown
        if self.coordinate.dist(ball.coordinate) <= self.KICK_RANGE:
            ball.kick(displacement, power)
            self._kick_cooldown = self.KICK_COOLDOWN_FRAMES
            return True
        return False
    
    def resolve_collision(self, other: _Player):
        """Push both players apart so they no longer overlap."""
        min_dist = self.RADIUS + other.RADIUS
        dist = self.coordinate.dist(other.coordinate)
        if dist >= min_dist or dist == 0:
            return

        # Direction from other → self
        overlap = (min_dist - dist) / 2
        norm = other.coordinate.vectorDistNormalized(self.coordinate)
        push = norm * overlap

        self.coordinate  += push
        other.coordinate -= push

    def resolve_ball_collision(self, ball: _Ball):
        """Push ball away and deflect its velocity on overlap."""
        min_dist = self.RADIUS + ball.RADIUS
        dist = self.coordinate.dist(ball.coordinate)
        if dist >= min_dist or dist == 0:
            return

        # Normal pointing from player centre → ball centre
        norm = self.coordinate.vectorDistNormalized(ball.coordinate)

        # Push ball fully outside the player
        overlap = min_dist - dist
        ball.coordinate += norm * overlap

        # Reflect the ball's velocity along the collision normal
        dot = ball.speed.x * norm.x + ball.speed.y * norm.y
        if dot < 0:                              # only reflect if moving toward player
            ball.speed.x -= 2 * dot * norm.x
            ball.speed.y -= 2 * dot * norm.y
    def is_colliding(self, other: "_Player") -> bool:
        return self.coordinate.dist(other.coordinate) < (self.RADIUS + other.RADIUS)


class _FSMOpponent(_Player):
    """Rule-based FSM opponent — mirrors the standalone script."""

    SHOOT_RANGE = 180
    MAX_SPEED = 4.0
    CLOSE_GOAL_FORCE = 14.0
    CLOSE_BALL_FORCE = 6.0

    def __init__(self, _x, _y, color, attack_goal_x, difficulty=1.0):
        super().__init__(_x, _y, color)
        self.enemyGoal = Vec2(attack_goal_x, HEIGHT//2)
        self.myGoal = Vec2(WIDTH - attack_goal_x, HEIGHT//2)
        self.difficulty = difficulty

    def act(self, ball: _Ball):
        if random.random() > self.difficulty:
            return
        bd         = self.coordinate.dist(ball.coordinate)
        near_ball  = bd < self.KICK_RANGE
        near_goal  = ball.coordinate.dist(self.enemyGoal) < self.SHOOT_RANGE
        ball_behind = abs(ball.coordinate.x - self.myGoal.x) < abs(self.coordinate.x - self.myGoal.x)

        if near_ball and near_goal:                          # SHOOT
            noise_y = random.uniform(-20, 20)
            displacement = Vec2(self.enemyGoal.x - ball.coordinate.x, self.enemyGoal.y - ball.coordinate.y + noise_y)
            self.try_kick(ball, displacement, self.CLOSE_GOAL_FORCE*self.difficulty)
        elif near_ball:                                      # DRIBBLE
            self.move(Vec2(ball.coordinate.x - self.coordinate.x, ball.coordinate.y - self.coordinate.y), self.MAX_SPEED*self.difficulty)
            self.try_kick(ball, Vec2(self.enemyGoal.x - ball.coordinate.x, self.enemyGoal.y - ball.coordinate.y), self.CLOSE_BALL_FORCE*self.difficulty)
        elif ball_behind:                                    # DEFEND
            mid = (self.myGoal + ball.coordinate) / 2
            self.move(Vec2(mid.x - self.coordinate.x, mid.y - self.coordinate.y), self.MAX_SPEED*self.difficulty)
        else:                                                # CHASE
            self.move(Vec2(ball.coordinate.x - self.coordinate.x, ball.coordinate.y - self.coordinate.y), self.MAX_SPEED*self.difficulty)


# ──────────────────────────────────────────────
# The Gymnasium Environment
# ──────────────────────────────────────────────
class SoccerEnv(gym.Env):
    """
    Single-agent Gymnasium environment.

    The *controlled* agent (red) plays on the left and attacks the right goal.
    The *opponent*  (blue) is driven by the FSM from the standalone script.

    ┌─────────────────────────────────────────────────────────────────────┐
    │ OBSERVATION SPACE  (14 floats, normalised)                          │
    ├────┬──────────────────┬─────────────────────────────────────────────┤
    │  0 │ agent_x          │ agent x ÷ WIDTH                             │
    │  1 │ agent_y          │ agent y ÷ HEIGHT                            │
    │  2 │ ball_x           │ ball x  ÷ WIDTH                             │
    │  3 │ ball_y           │ ball y  ÷ HEIGHT                            │
    │  4 │ ball_vx          │ clipped to ±1  (÷20)                        │
    │  5 │ ball_vy          │ clipped to ±1  (÷20)                        │
    │  6 │ opp_x            │ opponent x ÷ WIDTH                          │
    │  7 │ opp_y            │ opponent y ÷ HEIGHT                         │
    │  8 │ dist_agent_ball  │ ÷ max_dist  ∈ [0,1]                        │
    │  9 │ dist_opp_ball    │ ÷ max_dist  ∈ [0,1]                        │
    │ 10 │ dist_ball_goal   │ ÷ max_dist  ∈ [0,1]                        │
    │ 11 │ angle_to_goal    │ atan2 ÷ π  ∈ [-1,1]                        │
    │ 12 │ score_agent      │ clamped 0-5, ÷5                             │
    │ 13 │ score_opp        │ clamped 0-5, ÷5                             │
    └────┴──────────────────┴─────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │ ACTION SPACE  (Discrete 9)                                          │
    ├───┬─────────────────────────────────────────────────────────────────┤
    │ 0 │ stay                                                            │
    │ 1 │ move up                                                         │
    │ 2 │ move down                                                       │
    │ 3 │ move left                                                       │
    │ 4 │ move right                                                      │
    │ 5 │ move up-right + kick toward goal                                │
    │ 6 │ move down-right + kick toward goal                              │
    │ 7 │ kick toward goal (no movement)                                  │
    │ 8 │ clear / kick away from own goal                                 │
    └───┴─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │ REWARD STRUCTURE                                                    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ +10    scoring a goal                                               │
    │ -10    conceding a goal                                             │
    │ +0.05  per step closer to ball than opponent  (possession)          │
    │ +0.10  per step while ball is in opponent half                      │
    │ -0.01  per step                               (time penalty)        │
    └─────────────────────────────────────────────────────────────────────┘
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}
    _MAX_DIST = math.hypot(WIDTH, HEIGHT)

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        # ── Spaces ──────────────────────────────────────────────────────────
        low  = np.full(18, -1.0, dtype=np.float32)
        high = np.full(18,  1.0, dtype=np.float32)
        for i in (8, 9, 10, 12, 13, 14, 16, 17):   # distance / score features are ≥ 0
            low[i] = 0.0
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space      = spaces.MultiDiscrete([5, 3])

        # ── Internal state ───────────────────────────────────────────────────
        self.ball     : Optional[_Ball]        = None
        self.agent    : Optional[_Player]      = None
        self.opponent : Optional[_FSMOpponent] = None
        self._score_agent = 0
        self._score_opp   = 0
        self._step_count  = 0

        # ── Pygame handles ───────────────────────────────────────────────────
        self.screen : Optional[pygame.Surface]    = None
        self.clock  : Optional[pygame.time.Clock] = None
        self._font  : Optional[pygame.font.Font]  = None

        self.prev_a2b = None
        self.prev_b2g = None

    # ── Gym API ─────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.ball     = _Ball()
        self.agent    = _Player(225, HEIGHT // 2, RED)
        self.opponent = _FSMOpponent(675, HEIGHT // 2, BLUE, attack_goal_x=0)
        self._step_count  = 0
        self.prev_a2b = None
        self.prev_b2g = None

        if self.render_mode == "human":
            self._init_pygame()

        return self._get_obs(), {}

    def step(self, action: int):
        assert self.ball is not None, "Call reset() before step()."
        self._step_count += 1

        # 1. Agent action
        didAgentKick = self._apply_action(action)

        # 2. Opponent acts
        self.opponent.act(self.ball)

        # 3. Physics
        self.agent.tick_cooldown()
        self.opponent.tick_cooldown()
        self.ball.step()
        self.agent.resolve_collision(self.opponent)
        self.agent.resolve_ball_collision(self.ball)
        self.opponent.resolve_ball_collision(self.ball)

        # 4. Reward + termination
        reward     = -0.03
        terminated = False
        gt         = (HEIGHT - GOAL_HEIGHT) // 2
        gb         = gt + GOAL_HEIGHT

        if self.ball.coordinate.x > WIDTH and gt < self.ball.coordinate.y < gb:
            # Agent scored — ball entered RIGHT goal, which is opponent's goal
            reward              += 10.0
            self._score_agent   += 1
            terminated           = True

        elif self.ball.coordinate.x < 0 and gt < self.ball.coordinate.y < gb:
            # Opponent scored
            reward             -= 10.0
            self._score_opp    += 1
            terminated          = True

        # Shaping
        a2b = self.agent.coordinate.dist(self.ball.coordinate)
        o2b = self.opponent.coordinate.dist(self.ball.coordinate)
        b2g = self.ball.coordinate.dist(Vec2(WIDTH, HEIGHT / 2))

        if (self.prev_a2b != None):
            reward += 0.08 * (self.prev_a2b - a2b)

        if (self.prev_b2g != None):
            reward += 0.08 * (self.prev_b2g - b2g)

        self.prev_a2b = a2b
        self.prev_b2g = b2g

        if a2b < o2b:
            reward += 0.05
        if self.ball.coordinate.x > WIDTH / 2:
            reward += 0.10
        
        if self.agent.is_colliding(self.opponent):
            reward -= 0.05

        if didAgentKick:
            reward += 0.5

        truncated = (self._step_count >= MAX_STEPS)

        if self.render_mode == "human":
            self._render_frame()

        info = {
            "score_agent": self._score_agent,
            "score_opp":   self._score_opp,
            "step":        self._step_count,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame(return_rgb=True)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _apply_action(self, action) -> bool:
        move_action, kick_action = int(action[0]), int(action[1])
        goal_x = WIDTH
        goal_y = HEIGHT // 2

        move_map = {
            0: Vec2( 0,  0),   # stay
            1: Vec2( 0, -1),   # up
            2: Vec2( 0,  1),   # down
            3: Vec2(-1,  0),   # left
            4: Vec2( 1,  0),   # right
        }
        displacement = move_map[move_action]
        if displacement.x != 0 or displacement.y != 0:
            self.agent.move(displacement)

        if kick_action == 1:               # kick toward goal
            noise_y = random.uniform(-15, 15)
            return self.agent.try_kick(
                self.ball,
                Vec2(goal_x - self.ball.coordinate.x,
                    goal_y + noise_y - self.ball.coordinate.y),
                power=13.0
            )
        elif kick_action == 2:             # clear
            return self.agent.try_kick(
                self.ball,
                Vec2(WIDTH - self.ball.coordinate.x, goal_y - self.ball.coordinate.y),
                power=12.0
            )
        return False

    def _get_obs(self) -> np.ndarray:
        b  = self.ball
        a  = self.agent
        o  = self.opponent
        gx = float(WIDTH)
        gy = HEIGHT / 2.0

        angle = math.atan2(gy - a.coordinate.y, gx - a.coordinate.x) / math.pi
        ball_goal_angle = math.atan2(gy - b.coordinate.y, gx - b.coordinate.x) / math.pi

        return np.array([
            a.coordinate.x / WIDTH,
            a.coordinate.y / HEIGHT,
            b.coordinate.x / WIDTH,
            b.coordinate.y / HEIGHT,
            float(np.clip(b.speed.x / 20.0, -1, 1)),
            float(np.clip(b.speed.y / 20.0, -1, 1)),
            o.coordinate.x / WIDTH,
            o.coordinate.y / HEIGHT,
            a.coordinate.dist(b.coordinate) / self._MAX_DIST,
            o.coordinate.dist(b.coordinate) / self._MAX_DIST,
            b.coordinate.dist(Vec2(gx, gy)) / self._MAX_DIST,
            float(angle),
            min(self._score_agent, 5) / 5.0,
            min(self._score_opp,   5) / 5.0,
            self.agent._kick_cooldown / self.agent.KICK_COOLDOWN_FRAMES,
            float(ball_goal_angle),
            float(self.opponent.can_kick),
            math.hypot(b.speed.x, b.speed.y) / _Ball.MAX_SPEED,
        ], dtype=np.float32)

    # ── Pygame rendering ─────────────────────────────────────────────────────

    def _init_pygame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("SoccerEnv — RL Agent")
            self.clock  = pygame.time.Clock()
            self._font  = pygame.font.SysFont(None, 32)

    def _render_frame(self, return_rgb: bool = False):
        if self.screen is None:
            self._init_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        surf = self.screen

        # ── Field ────────────────────────────────
        surf.fill(GREEN)
        pygame.draw.line(surf, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)
        pygame.draw.circle(surf, WHITE, (WIDTH // 2, HEIGHT // 2), 60, 2)
        gt = (HEIGHT - GOAL_HEIGHT) // 2
        pygame.draw.rect(surf, WHITE, (0,         gt, 8, GOAL_HEIGHT))
        pygame.draw.rect(surf, WHITE, (WIDTH - 8, gt, 8, GOAL_HEIGHT))

        # ── Entities ─────────────────────────────
        self.ball.draw(surf)
        self.agent.draw(surf)
        self.opponent.draw(surf)

        # ── HUD ──────────────────────────────────
        score_txt = self._font.render(
            f"step {self._step_count}",
            True, WHITE,
        )
        surf.blit(score_txt, (WIDTH // 2 - score_txt.get_width() // 2, 8))

        agent_txt = self._font.render(f"Agent: {self._score_agent}", True, RED)

        surf.blit(agent_txt, (8, 8))

        opp_txt = self._font.render(f"FSM: {self._score_opp}", True, BLUE)
        
        surf.blit(opp_txt, (WIDTH - opp_txt.get_width() - 8, 8))

        pygame.display.flip()
        if self.clock:
            self.clock.tick(FPS)

        if return_rgb:
            return np.transpose(pygame.surfarray.array3d(surf), axes=(1, 0, 2))
        return None


# ──────────────────────────────────────────────
# Run this file directly to train + watch
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    print("=== Checking environment …")
    train_env = SoccerEnv(render_mode=None)
    check_env(train_env, warn=True)
    print("=== Environment OK.\n")

    print("=== Training PPO for 200 000 steps …")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        ent_coef=0.05,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        clip_range=0.15,
        learning_rate=1e-4,
        tensorboard_log="./soccer_tb/",
        device="cpu"
    )
    for stage, (difficulty, steps) in enumerate([
        (0.2, 1_500_000),
        (0.5, 1_500_000),
        (0.8, 2_000_000),
    ]):
        train_env.reset()
        train_env.opponent.difficulty = difficulty
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
    model.save("soccer_ppo")
    print("=== Model saved to soccer_ppo.zip\n")

    print("=== Watching trained agent …")
    eval_env = SoccerEnv(render_mode="human")
    obs, _   = eval_env.reset()
    eval_env.opponent.difficulty = 0.8
    for _ in range(20_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = eval_env.step(action)
        if terminated or truncated:
            print(info)
            obs, _ = eval_env.reset()
            eval_env.opponent.difficulty = 0.8
    eval_env.close()