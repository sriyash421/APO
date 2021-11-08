import json
import itertools
import os

def read_config():
    with open("master_config.json", 'r') as json_file: 
      data = json.load(json_file)
    return data

def main():
    config = read_config()
    try:
      assert os.path.exists("cfg_temp/")
    except:
      os.makedirs('cfg_temp/')
    i=0
    for param_vals in itertools.product(*config.values()):
        cfg = dict(zip(config.keys(), param_vals))
        config_struct ={
            "env" : cfg["env"],
            "decay" : cfg["decay"],
            "alpha" : cfg["alpha"],
            "v" : cfg["v"],
            "seed" : cfg["seed"],
        }
        alpha = str(config_struct["alpha"])
        v = str(config_struct['v'])
        decay = str(config_struct['decay'])
        env = str(config_struct['env'])
        seed = str(config_struct['seed'])
        train_cmd = "python -m tonic.train --header \"import apo\" --agent \"apo.APO(alpha="+ alpha +", v="+ v +", trace_decay="+ decay +")\" --environment \"tonic.environments.Gym(\'"+ env +"\')\" --trainer \"tonic.Trainer(test_episodes=10, epoch_steps=int(2e3), steps=int(3e6))\" --seed "+seed+" --name \"APO-"+env+"-V_"+v+"-Alpha_"+alpha+"-TraceDecay_"+decay+"\""
        w_name =  str(i) + '.sh'
        i = i+1
        text_file = open(w_name, "w")
        n = text_file.write(train_cmd)
        text_file.close()
        # jsonstr = json.dumps(config_struct)
        # w_name = 'cfg_temp/'+ str(i) + '.json'
        # i = i+1
        # with open(w_name, "w") as outfile: 
        #     outfile.write(jsonstr) 


if __name__ == '__main__':
    main()
