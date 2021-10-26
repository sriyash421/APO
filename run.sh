#!/bin/bash

cd results

python3 -m tonic.train \
--header "import apo" \
--agent "apo.APO()" \
--environment "tonic.environments.Gym('HalfCheetah-v2')" \
--seed 0