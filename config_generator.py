import json
import itertools
import os


def read_config(agent):
    with open(f"master_config_{agent}.json", 'r') as json_file:
        data = json.load(json_file)
    return data


def main():
    i = 0
    try:
        assert os.path.exists("cfg_temp/")
    except:
        os.makedirs('cfg_temp/')

    config = read_config('apo')
    for param_vals in itertools.product(*config.values()):
        cfg = dict(zip(config.keys(), param_vals))
        alpha = str(cfg["alpha"])
        v = str(cfg['v'])
        decay = str(cfg['decay'])
        env = str(cfg['env'])
        seed = str(cfg['seed'])
        train_cmd = f"python -m tonic.train --header 'import apo' \
            --agent 'apo.APO(alpha={alpha}, v={v}, trace_decay={decay})' \
            --environment 'tonic.environments.Gym(\'{env}\')' \
            --trainer 'tonic.Trainer(test_episodes=10, epoch_steps=int(2e3), steps=int(3e6))' \
            --seed {seed} \
            --name 'APO-v{v}-a{alpha}-t{decay}'"
        w_name = str(i) + '.sh'
        i = i+1
        text_file = open(w_name, "w")
        n = text_file.write(train_cmd)
        text_file.close()
        # jsonstr = json.dumps(config_struct)
        # w_name = 'cfg_temp/'+ str(i) + '.json'
        # i = i+1
        # with open(w_name, "w") as outfile:
        #     outfile.write(jsonstr)

    config = read_config('ppo')
    for param_vals in itertools.product(*config.values()):
        cfg = dict(zip(config.keys(), param_vals))
        discount_factor = str(cfg["discount_factor"])
        decay = str(cfg['decay'])
        env = str(cfg['env'])
        seed = str(cfg['seed'])
        train_cmd = f"python -m tonic.train --header 'import apo' \
            --agent 'apo.PPO(discount_factor={discount_factor}, trace_decay={decay})' \
            --environment 'tonic.environments.Gym(\'{env}\')' \
            --trainer 'tonic.Trainer(test_episodes=10, epoch_steps=int(2e3), steps=int(3e6))' \
            --seed {seed} \
            --name 'APO-d{discount_factor}-t{decay}'"
        w_name = str(i) + '.sh'
        i = i+1
        text_file = open(w_name, "w")
        n = text_file.write(train_cmd)
        text_file.close()


if __name__ == '__main__':
    main()
