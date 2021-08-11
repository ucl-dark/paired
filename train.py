import sys
import os
import time
import timeit
import logging
import signal
from arguments import parser

import torch
import gym
import matplotlib as mpl
# mpl.use("macOSX")
import matplotlib.pyplot as plt

from baselines.common.vec_env import VecNormalize
from baselines.logger import HumanOutputFormat

from envs.multigrid import *
from envs.multigrid.adversarial import *
from envs.minihack.adversarial import *

from envs.runners.adversarial_runner import AdversarialRunner
from util import make_agent, FileWriter, safe_checkpoint, create_parallel_env
from eval import Evaluator


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"

    args = parser.parse_args()
    
    # === Configure logging ===
    if args.xpid is None:
        args.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.expandvars(os.path.expanduser(args.log_dir))
    filewriter = FileWriter(
        xpid=args.xpid, xp_args=args.__dict__, rootdir=log_dir
    )
    screenshot_dir = os.path.join(log_dir, args.xpid, 'screenshots')
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir, exist_ok=True)

    def log_stats(stats):
        filewriter.log(stats)
        if args.verbose:
            HumanOutputFormat(sys.stdout).writekvs(stats)

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    # === Determine device ====
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if 'cuda' in device.type:
        torch.backends.cudnn.benchmark = True
        print('Using CUDA\n')

    # === Create parallel envs ===
    venv, ued_venv = create_parallel_env(args)

    is_training_env = args.ued_algo in ['minimax', 'paired', 'flexible_paired']
    is_paired = args.ued_algo in ['paired', 'flexible_paired']

    agent = make_agent(name='agent', env=venv, args=args, device=device)
    adversary_agent, adversary_env = None, None
    if is_paired:
        adversary_agent = make_agent(name='adversary_agent', env=venv, args=args, device=device)

    if is_training_env:
        adversary_env = make_agent(name='adversary_env', env=venv, args=args, device=device)

    # === Signal handler ===
    def signal_handler(sig, frame):
        venv.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # === Create runner ===
    train_runner = AdversarialRunner(
        args=args,
        agent=agent,
        venv=venv,
        ued_venv=ued_venv,
        adversary_agent=adversary_agent, 
        adversary_env=adversary_env, 
        train=True,
        device=device)

    # === Configure checkpointing ===
    timer = timeit.default_timer
    last_checkpoint_time = None
    initial_update_count = 0
    last_logged_update_at_restart = -1
    checkpoint_path = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (log_dir, args.xpid, "model.tar"))
    )

    def checkpoint(index=None):
        if args.disable_checkpoint:
            return
        safe_checkpoint({'runner_state_dict': train_runner.state_dict()}, 
                        checkpoint_path,
                        index=index, 
                        archive_interval=args.archive_interval)
        logging.info("Saved checkpoint to %s", checkpoint_path)


    # === Load checkpoint ===
    if args.checkpoint and os.path.exists(checkpoint_path):
        checkpoint_states = torch.load(checkpoint_path)
        last_logged_update_at_restart = filewriter.latest_tick() # ticks are 0-indexed updates
        train_runner.load_state_dict(checkpoint_states['runner_state_dict'])
        initial_update_count = train_runner.num_updates
        logging.info(f"Resuming preempted job after {initial_update_count} updates\n") # 0-indexed next update

    # Set up Evaluator
    evaluator = None
    if args.test_env_names:
        test_envs = args.test_env_names.split(',')
        evaluator = Evaluator(
            test_envs, 
            num_processes=args.test_num_processes, 
            num_episodes=args.test_num_episodes,
            device=device)

    # === Train ===
    update_start_time = timer()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    steps_previous = 0
    for j in range(initial_update_count, num_updates):
        stats = train_runner.run()

        # === Perform logging ===
        if train_runner.num_updates <= last_logged_update_at_restart:
            continue

        log = (j % args.log_interval == 0) or j == num_updates - 1
        save_screenshot = \
            args.screenshot_interval > 0 and \
                (j % args.screenshot_interval == 0)

        if log:
            # Eval
            test_stats = {}
            if evaluator is not None and (j % args.test_interval == 0 or j == num_updates - 1):
                test_stats = evaluator.evaluate(train_runner.agents['agent'])
                stats.update(test_stats)
            else:
                stats.update({k: None for k in evaluator.stats_keys})

            update_end_time = timer()
            num_incremental_updates = 1 if j == 0 else args.log_interval
            sps = num_incremental_updates*(args.num_processes * args.num_steps) / (update_end_time - update_start_time)
            update_start_time = update_end_time
            stats.update({'sps': sps})
            stats.update(test_stats)
            log_stats(stats)

        if last_checkpoint_time is None:
            last_checkpoint_time = timer()
        if j == num_updates - 1 or \
            (args.save_interval > 0 and timer() - last_checkpoint_time > args.save_interval * 60):
            checkpoint(train_runner.num_updates)
            last_checkpoint_time = timer()
            logging.info(f"\nSaved checkpoint after update {j + 1}")
        elif train_runner.num_updates > 0 and args.archive_interval > 0 \
            and train_runner.num_updates % args.archive_interval == 0:
            checkpoint(train_runner.num_updates)
            last_checkpoint_time = timer()
            logging.info(f"\nSaved checkpoint after update {j + 1}")

        if save_screenshot:
            venv.reset_agent()
            images = venv.get_images()
            plt.axis('off')
            plt.imshow(images[0])
            plt.savefig(os.path.join(screenshot_dir, f'update_{j}.png'), bbox_inches='tight')
            plt.close()

            if args.env_name.startswith('MiniHack'):
                # ASCII obs
                with open(os.path.join(screenshot_dir, f'update_{j}.txt'), 'w+') as fout:
                    fout.write(venv.get_grid_str())

                # des file
                with open(os.path.join(screenshot_dir, f'update_{j}.des'), 'w+') as fout:
                    fout.write(venv.get_des_file())

    evaluator.close()
    venv.close()
