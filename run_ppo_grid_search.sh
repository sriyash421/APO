#!/bin/bash

# cd results

for seed in {0..5}
do  
    for env in HalfCheetah-v3 Ant-v3 Swimmer-v3
    do
        for decay in 0.8 0.9 0.95 0.99
        do
            for discount in 0.9 0.95 0.99 0.999
            do
                python3 -m tonic.train \
                --header "import apo" \
                --agent "apo.PPO(discount_factor=${discount}, trace_decay=${decay})" \
                --environment "tonic.environments.Gym('${env}')" \
                --trainer "tonic.Trainer(test_episodes=10, epoch_steps=int(2e3), steps=int(3e6))" \
                --seed $seed \
                --name "PPO-${env}-Discount_${discount}-TraceDecay_${decay}" &
            done
        done
    done
done