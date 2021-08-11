import argparse
import json
import os


def generate_train_cmds(
    params, num_trials=1, start_index=0, newlines=False, 
    xpid_generator=None, xpid_prefix=''):
    separator = ' \\\n' if newlines else ' '

    cmds = []

    if xpid_generator:
        params['xpid'] = xpid_generator(params, xpid_prefix)

    start_seed = params['seed']

    for t in range(num_trials):
        params['seed'] = start_seed + t + start_index

        cmd = [f'python -m train']

        trial_idx = t + start_index
        for k, v in params.items():
            if k == 'xpid':
                v = f'{v}_{trial_idx}'

            cmd.append(f'--{k}={v}')

        cmd = separator.join(cmd)

        cmds.append(cmd)

    return cmds


def generate_all_params_for_grid(grid, defaults={}):
    def update_params_with_choices(prev_params, param, choices):
        updated_params = []
        for v in choices:
            for p in prev_params:
                updated = p.copy()
                updated[param] = v
                updated_params.append(updated)

        return updated_params

    all_params = [{}]
    for param, choices in grid.items():
        all_params = update_params_with_choices(all_params, param, choices)

    full_params = []
    for p in all_params:
        d = defaults.copy()
        d.update(p)
        full_params.append(d)

    return full_params


def parse_args():
    parser = argparse.ArgumentParser(description='Make commands')

    parser.add_argument(
        '--dir',
        type=str,
        default='train_scripts/configs/',
        help='Path to directory with .json configs')

    parser.add_argument(
        '--json',
        type=str,
        default=None,
        help='Name of .json config for hyperparameter search-grid')

    parser.add_argument(
        '--num_trials',
        type=int,
        default=1,
        help='Name of .json config for hyperparameter search-grid')

    parser.add_argument(
        '--start_index',
        default=0,
        type=int,
        help='Starting trial index of xpid runs')

    parser.add_argument(
        '--count',
        action='store_true',
        help='Print number of generated commands at the end of output.')

    parser.add_argument(
        "--checkpoint",
        action='store_true',
        help='Whether to start from checkpoint'
    )
    
    return parser.parse_args()


def xpid_from_params(p, prefix=''):
    ued_algo = p['ued_algo']
    is_train_env = ued_algo in ['paired', 'flexible_paired', 'minimax']

    env_prefix = p['env_name']

    prefix_str = '' if prefix == '' else f'-{prefix}'

    rnn_prefix = ''
    rnn_agent = 'a' if p['recurrent_agent'] else ''
    rnn_env = 'e' if p['recurrent_adversary_env'] and is_train_env else ''
    if rnn_agent or rnn_env:
        rnn_arch = p['recurrent_arch']
        rnn_hidden = p['recurrent_hidden_size']
        rnn_prefix = f'-{rnn_arch}{rnn_hidden}{rnn_agent}{rnn_env}'

    ppo_prefix = f"-lr{p['lr']}-epoch{p['ppo_epoch']}-mb{p['num_mini_batch']}-v{p['value_loss_coef']}"

    entropy_prefix = f"-henv{p['adv_entropy_coef']}-ha{p['entropy_coef']}"

    timelimits = '-tl' if p['handle_timelimits'] else ''

    return f'ued{prefix_str}-{env_prefix}-{ued_algo}{rnn_prefix}{ppo_prefix}{entropy_prefix}{timelimits}'

if __name__ == '__main__':
    args = parse_args()

    # Default parameters
    params = {
        'xpid': 'test',

        # Env params
        'env_name': 'MultiGrid-GoalLastAdversarial-v0',
        'use_gae': True,
        'gamma': 0.995,
        'gae_lambda': 0.95,
        'seed': 88,

        # Model params
        'recurrent_arch': 'lstm',
        'recurrent_agent': True,
        'recurrent_adversary_env': True,
        'recurrent_hidden_size': 256,

        # Learning params
        'lr': 1e-4,
        'num_steps': 256,  # unroll length
        'num_processes': 32,  # number of actor processes
        'num_env_steps': 1000000000,  # total training steps
        'ppo_epoch': 20,
        'num_mini_batch': 1,
        'entropy_coef': 0.,
        'value_loss_coef': 0.5,
        'clip_param': 0.2,
        'clip_value_loss': True,
        'adv_entropy_coef': 0.,
        'algo': 'ppo',
        'ued_algo': 'paired',

        # Logging params
        'log_interval': 1,
        'screenshot_interval': 1000,
        'log_grad_norm': False
    }

    json_filename = args.json
    if not json_filename.endswith('.json'):
        json_filename += '.json'

    grid_path = os.path.join(os.path.expandvars(os.path.expanduser(args.dir)), json_filename)
    config = json.load(open(grid_path))
    grid = config['grid']
    xpid_prefix = '' if 'xpid_prefix' not in config else config['xpid_prefix']

    if args.checkpoint:
        params['checkpoint'] = True

    # Generate all parameter combinations within grid, using defaults for fixed params
    all_params = generate_all_params_for_grid(grid, defaults=params)

    # Print all commands
    count = 0
    for p in all_params:
        cmds = generate_train_cmds(p,
            num_trials=args.num_trials, 
            start_index=args.start_index, 
            newlines=True, 
            xpid_generator=xpid_from_params, 
            xpid_prefix=xpid_prefix)

        for c in cmds:
            print(c + '\n')
            count += 1

    if args.count:
        print(f'Generated {count} commands.')