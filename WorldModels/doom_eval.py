import numpy as np
from env import make_env
from controller import make_controller, simulate
from utils import PARSER

ret_std_dict = {}

for p_i in ['WM', '0.05', '0.1', '0.2', '0.3', '0.5']:
    if p_i == 'WM':
        args = PARSER.parse_args(['--config_path', 'configs/wm_doom.config'])
        e = make_env(args, dream_env=False)
        c = make_controller(args)
        c.load_model('results/WorldModels/DoomTakeCover-v0/eval_input0.0_hidden0.0_log/DoomTakeCover-v0.cma.16.64.best.json')
    else:
        args = PARSER.parse_args(['--config_path', 'configs/ddl05_doom.config'])
        e = make_env(args, dream_env=False)
        c = make_controller(args)
        c.load_model('results/FullDropout05_WorldModels/DoomTakeCover-v0/eval_input{}_hidden{}_log/DoomTakeCover-v0.cma.16.64.best.json'.format(p_i, p_i))
    rets, _ = simulate(c, e, render_mode=False, num_episode=1000, seed=1)
    ret_std_dict[p_i] = (np.mean(rets), np.std(rets))
    print(p_i, np.mean(rets), np.std(rets))
print(ret_std_dict)
