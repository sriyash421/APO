#!/bin/bash

# cd results

for i in {0..5}
do

python3 -m tonic.train \
--header "import apo" \
--agent "apo.APO()" \
--environment "tonic.environments.Gym('HalfCheetah-v2')" \
--trainer "tonic.Trainer(test_episodes=10, epoch_steps=int(2e3), steps=int(3e6))" \
--seed $i &

python3 -m tonic.train \
--header "import apo" \
--agent "apo.PPO()" \
--environment "tonic.environments.Gym('HalfCheetah-v2')" \
--trainer "tonic.Trainer(test_episodes=10, epoch_steps=int(2e3), steps=int(3e6))" \
--seed $i &

wait

done