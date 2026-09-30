import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard


def reset_env(env):
    print("resetting environment...", flush=True)
    obs, info = env.reset()
    print("reset done", flush=True)
    return obs, info


from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name", "usb_insert", "Name of experiment corresponding to folder."
)
flags.DEFINE_integer(
    "successes_needed", 250, "Number of successful transistions to collect."
)


success_key = False
fail_key = False
reset_key = False


def on_press(key):
    global success_key
    global fail_key
    global reset_key
    if key == keyboard.Key.space:
        success_key = True
        return
    if key == keyboard.Key.esc:
        fail_key = True
        return
    try:
        if key.char in ("s", "S"):
            success_key = True
        elif key.char in ("f", "F"):
            fail_key = True
        elif key.char in ("r", "R"):
            reset_key = True
    except AttributeError:
        pass


def main(_):
    global success_key
    global fail_key
    global reset_key
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, _ = reset_env(env)
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)

    while len(successes) < success_needed:
        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]
        # print(obs["state"][0].shape)
        # assert(obs["state"][0].shape[1] == 20)
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        obs = next_obs
        if success_key:
            transition["rewards"] = 1.0
            transition["masks"] = 0.0
            transition["dones"] = True
            successes.append(transition)
            pbar.update(1)
            print(
                f"success recorded ({len(successes)}/{success_needed}); resetting",
                flush=True,
            )
            success_key = False
            obs, _ = reset_env(env)
        elif fail_key:
            transition["rewards"] = 0.0
            transition["masks"] = 0.0
            transition["dones"] = True
            failures.append(transition)
            print("failure recorded; resetting", flush=True)
            fail_key = False
            obs, _ = reset_env(env)
        elif reset_key:
            print("manual reset without recording", flush=True)
            reset_key = False
            obs, _ = reset_env(env)

        # if done or truncated:

    listener.stop()

    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = (
        f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
    )
    with open(file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"saved {success_needed} successful transitions to {file_name}")

    file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"saved {len(failures)} failure transitions to {file_name}")


if __name__ == "__main__":
    app.run(main)
