# 🧠 PPO 학습, 저장, 로딩, 시뮬레이션 시각화까지 전체 과정

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import time
import os

# ------------------------------------
# 1️⃣ 환경 생성 (렌더링은 없이 학습만)
env = gym.make("HalfCheetah-v4")  # 학습 시에는 render_mode 없음
vec_env = DummyVecEnv([lambda: env])  # SB3는 VecEnv 형태 필요
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)  # 관측값 정규화

# ------------------------------------
# 2️⃣ PPO 에이전트 정의 및 학습
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    ent_coef=0.01,          # 탐험을 더 하도록 유도
    tensorboard_log="./ppo_cheetah_tensorboard/"
)

model.learn(total_timesteps=500_000)  # 💡 충분히 긴 학습 시간 확보

# ------------------------------------
# 3️⃣ 모델 및 환경 저장
model.save("ppo_halfcheetah")
vec_env.save("ppo_halfcheetah_env.pkl")  # VecNormalize 파라미터 저장

# ------------------------------------
# 4️⃣ 시뮬레이션 보기용 환경 생성 (렌더링 포함)
eval_env = gym.make("HalfCheetah-v4", render_mode="human")
eval_vec_env = DummyVecEnv([lambda: eval_env])
eval_vec_env = VecNormalize.load("ppo_halfcheetah_env.pkl", eval_vec_env)
eval_vec_env.training = False  # 평가 모드로 설정
eval_vec_env.norm_reward = False

# ------------------------------------
# 5️⃣ 저장된 모델 불러오기 및 환경 세팅
loaded_model = PPO.load("ppo_halfcheetah")
loaded_model.set_env(eval_vec_env)

# ------------------------------------
# ▶️ 시뮬레이션 및 리워드 측정
obs = eval_vec_env.reset()
episode_reward = 0
episode_count = 0

for step in range(1000):
    action, _ = loaded_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_vec_env.step(action)

    # VecNormalize 환경이기 때문에 reward는 array로 나옴
    episode_reward += reward[0]

    time.sleep(0.01)

    if terminated or truncated:
        print(f"✅ Episode {episode_count + 1} reward: {episode_reward:.2f}")
        episode_reward = 0
        episode_count += 1
        obs = eval_vec_env.reset()
