import os
import sys
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name", "usb_insert", "Name of experiment corresponding to folder."
)
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

success_key = False
fail_key = False


def clear_manual_keys():
    global success_key, fail_key
    success_key = False
    fail_key = False


def on_press(key):
    global success_key, fail_key
    if key == keyboard.Key.space:
        success_key = True
    elif key == keyboard.Key.esc:
        fail_key = True
    else:
        try:
            if key.char == "s":
                success_key = True
            elif key.char == "f":
                fail_key = True
        except AttributeError:
            pass


def main(_):
    global success_key, fail_key
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    print(FLAGS.exp_name)
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, info = env.reset()
    clear_manual_keys()
    print("Reset done")
    print(
        "Manual demo labels: SPACE = save current trajectory as success, f or ESC = discard current trajectory and reset."
    )
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0

    while success_count < success_needed:
        # print("czech")
        actions = np.zeros(env.action_space.sample().shape)
        # print(actions)
        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew
        if "intervene_action" in info:
            actions = info["intervene_action"]
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
        )
        trajectory.append(transition)

        pbar.set_description(f"Return: {returns}")
        # print("\n\n\n REWARD IS:", rew, "\n\n\n")
        obs = next_obs
        if success_key:
            if trajectory:
                trajectory[-1]["rewards"] = 1.0
                trajectory[-1]["masks"] = 0.0
                trajectory[-1]["dones"] = True
                trajectory[-1]["infos"] = copy.deepcopy(trajectory[-1].get("infos", {}))
                trajectory[-1]["infos"]["succeed"] = True
            for transition in trajectory:
                transitions.append(copy.deepcopy(transition))
            success_count += 1
            pbar.update(1)
            print(f"success demo recorded ({success_count}/{success_needed})")
            success_key = False
            trajectory = []
            returns = 0
            obs, info = env.reset()
            clear_manual_keys()
        elif fail_key:
            print("failure marked; discarded current trajectory")
            fail_key = False
            trajectory = []
            returns = 0
            obs, info = env.reset()
            clear_manual_keys()
        elif done or truncated:
            print("episode timed out/truncated; discarded current trajectory")
            trajectory = []
            returns = 0
            obs, info = env.reset()
            clear_manual_keys()

    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}", flush=True)

    pbar.close()
    listener.stop()
    try:
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "img_queue"):
            unwrapped.img_queue.put(None)
        if hasattr(unwrapped, "close_cameras"):
            unwrapped.close_cameras()
    except Exception as exc:
        print(f"failed to close cameras cleanly: {exc}", flush=True)
    env.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    app.run(main)
