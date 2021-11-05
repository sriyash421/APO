#!/bin/bash

# cd results

for i in {0..2}
do

python3 -m tonic.train \
--header "import apo" \
--agent "apo.APO()" \
--environment "tonic.environments.Gym('HalfCheetah-v2')" \
--seed $i &

python3 -m tonic.train \
--header "import apo" \
--agent "apo.PPO()" \
--environment "tonic.environments.Gym('HalfCheetah-v2')" \
--seed $i &

wait

done