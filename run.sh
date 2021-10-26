#!/bin/bash

python3 -m tonic.train \
--header "import apo" \
--agent "apo.APO()" \
--environment "tonic.environments.Gym('HalfCheetah-v3')" \
--seed 0