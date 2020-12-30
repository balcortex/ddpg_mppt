import os

import matlab.engine

from src.behavioral_cloning import BehavioralCloning
from src.experience import ExperienceSourceDiscountedSteps
from src.policies import DDPGPolicy, PerturbObservePolicy
from src.pv_array import PVArray
from src.pv_env import PVEnv
from src.reward import RewardPowerDeltaPower
from src.utils import read_weather_csv, save_dict

PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
PV_CKP_PATH = os.path.join("data", "01_pvarray_iv.json")
WEATHER_TRAIN_PATH = os.path.join("data", "weather_real_train.csv")
WEATHER_VAL_PATH = os.path.join("data", "weather_real_val.csv")
WEATHER_TEST_PATH = os.path.join("data", "weather_real_test.csv")
WEATHER_SIM_PATH = os.path.join("data", "weather_sim.csv")
EXPERIMENT_PATH = os.path.join("experiments", "bc")
STATES = ["v_norm", "p_norm", "dv_norm", "dp_norm"]
V_INITIAL = 22

GAMMA = 0.5
N_STEPS = 1
DEMO_BATCH_SIZE = 512
DEMO_BUFFER_SIZE = 5_000
# EPOCHS = 5_000
EPOCHS = 100_000
ACTOR_LR = 1e-4
# ACTOR_L2 = 1e-3
ACTOR_L2 = 1e-3

try:
    engine.quit()
except NameError:
    pass
engine = matlab.engine.connect_matlab()


def get_envs():
    reward_fn = RewardPowerDeltaPower(norm=True)
    pvarray = PVArray.from_json(
        path=PV_PARAMS_PATH,
        engine=engine,
        ckp_path=PV_CKP_PATH,
        f_precision=1,
    )
    weather_df = read_weather_csv(WEATHER_TRAIN_PATH)
    weather_val_df = read_weather_csv(WEATHER_VAL_PATH)
    weather_test_df = read_weather_csv(WEATHER_TEST_PATH)
    demo_env = PVEnv(
        pvarray=pvarray,
        weather_df=weather_df,
        states=STATES,
        reward_fn=reward_fn,
        max_steps=500,
    )
    val_env = PVEnv(
        pvarray=pvarray,
        weather_df=weather_val_df,
        states=STATES,
        reward_fn=reward_fn,
        v0=12,
        deterministic=True,
    )
    test_env = PVEnv(
        pvarray=pvarray,
        weather_df=weather_test_df,
        states=STATES,
        reward_fn=reward_fn,
        v0=12,
        deterministic=True,
    )

    return demo_env, val_env, test_env


class Experiment:
    def __init__(
        self,
        gamma: float,
        n_steps: int,
        demo_batch_size: int,
        demo_buffer_size: int,
        actor_lr: float,
        actor_l2: float,
        load_checkpoint: str = None,
    ):
        self.gamma = gamma
        self.n_steps = n_steps
        self.demo_batch_size = demo_batch_size
        self.demo_buffer_size = demo_buffer_size
        self.actor_lr = actor_lr
        self.actor_l2 = actor_l2
        self.load_checkpoint = load_checkpoint

        name = "exp" + str(len(os.listdir(EXPERIMENT_PATH)) // 4).zfill(4)
        self.path = os.path.join(EXPERIMENT_PATH, name)

        self.demo_env, self.val_env, self.test_env = get_envs()
        actor = BehavioralCloning.create_actor(self.demo_env)

        val_policy = DDPGPolicy(env=self.val_env, net=actor, noise=None, schedule=None)
        test_policy = DDPGPolicy(
            env=self.test_env, net=actor, noise=None, schedule=None
        )
        po_policy = PerturbObservePolicy(
            env=self.demo_env, v_step=0.2, dv_index="dv", dp_index="dp", noise=None
        )

        self.val_exp_source = ExperienceSourceDiscountedSteps(
            policy=val_policy, gamma=gamma, n_steps=n_steps, steps=1
        )
        self.test_exp_source = ExperienceSourceDiscountedSteps(
            policy=test_policy, gamma=gamma, n_steps=n_steps, steps=1
        )
        self.po_exp_source = ExperienceSourceDiscountedSteps(
            policy=po_policy, gamma=gamma, n_steps=n_steps, steps=1
        )

        self.agent = BehavioralCloning(
            demo_exp_source=self.po_exp_source,
            test_exp_sorce=self.test_exp_source,
            actor=actor,
            demo_buffer_size=DEMO_BUFFER_SIZE,
            demo_batch_size=DEMO_BATCH_SIZE,
            actor_lr=ACTOR_LR,
            actor_l2=ACTOR_L2,
            load_checkpoint=load_checkpoint,
        )

    def run(self, epochs: int = 10_000, val_every: int = 1_000) -> None:
        iters = epochs // val_every
        eff = 0

        for _ in range(iters):
            self.agent.learn(epochs=epochs // iters, train_steps=1)
            eff_ = self.run_val(episodes=1)

            if eff_ < eff:
                break

            eff = eff_

    def save(self) -> None:
        dic_path = self.path + ".json"

        dic = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "demo_batch_size": self.demo_batch_size,
            "demo_buffer_size": self.demo_buffer_size,
            "actor_lr": self.actor_lr,
            "actor_l2": self.actor_l2,
        }
        save_dict(dic, dic_path)
        self.save_agent(self.path + ".tar")

        self.test_exp_source.play_episode()
        self.test_env.render_vs_true(show=False, save_path=self.path)

    def run_val(self, episodes: int = 1) -> float:
        eff = 0
        for _ in range(episodes):
            self.val_exp_source.play_episode()
            p_real, v_real, _ = self.val_env.pvarray.get_true_mpp(
                self.val_env.history.g, self.val_env.history.t
            )
            eff += PVArray.mppt_eff(p_real, self.val_env.history.p)

        eff /= episodes
        return eff

    def test(self, episodes: int = 1) -> float:
        eff = 0
        for _ in range(episodes):
            self.test_exp_source.play_episode()
            p_real, v_real, _ = self.test_env.pvarray.get_true_mpp(
                self.test_env.history.g, self.test_env.history.t
            )
            eff += PVArray.mppt_eff(p_real, self.test_env.history.p)

        eff /= episodes
        return eff

    def save_agent(self, path: str) -> None:
        self.agent.save(path)


max_test_eff = 0
# for ACTOR_L2 in [5e-4, 4.5e-4, 4e-4]:
for _ in range(50):
    experiment = Experiment(
        gamma=GAMMA,
        n_steps=N_STEPS,
        demo_batch_size=DEMO_BATCH_SIZE,
        demo_buffer_size=DEMO_BUFFER_SIZE,
        actor_lr=ACTOR_LR,
        actor_l2=ACTOR_L2,
        load_checkpoint=None,
    )

    # experiment.po_exp_source.play_episode()
    # experiment.demo_env.render_vs_true(label="PO")

    experiment.run(epochs=EPOCHS, val_every=1000)
    experiment.save()
    eff = experiment.test(episodes=1)

    if eff > max_test_eff:
        max_test_eff = eff
        experiment.save_agent(os.path.join(EXPERIMENT_PATH, "best.tar"))

# Testing
experiment = Experiment(
    gamma=GAMMA,
    n_steps=N_STEPS,
    demo_batch_size=DEMO_BATCH_SIZE,
    demo_buffer_size=DEMO_BUFFER_SIZE,
    actor_lr=ACTOR_LR,
    actor_l2=ACTOR_L2,
    load_checkpoint=os.path.join(EXPERIMENT_PATH, "best.tar"),
)

# Re-run the best experiment
eff = experiment.test()
experiment.save()
print(eff)


# Compare RL to PO
experiment.test_exp_source.play_episode()
experiment.test_env.render_vs_true()

po_policy = PerturbObservePolicy(
    env=experiment.test_env, v_step=0.2, dv_index="dv", dp_index="dp", noise=None
)

po_exp_source = ExperienceSourceDiscountedSteps(
    policy=po_policy, gamma=GAMMA, n_steps=N_STEPS, steps=1
)
po_exp_source.play_episode()
experiment.test_env.render_vs_true(label="PO")
