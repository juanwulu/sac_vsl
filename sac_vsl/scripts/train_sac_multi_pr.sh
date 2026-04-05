#!/usr/bin/bash
#
# Copyright (c) 2022 - , Juanwu Lu. All rights reserved.
#
# File: train_sac_multi_pr.sh
# Description: Run SAC on I80 VSL with multiple penetration rate settings.
# Author: Juanwu Lu
# Date: Dec-2-22

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd );

echo "Running SAC on I80 VSL with multiple penetration rate settings...";
for p_rate in 0.1 0.2 0.5 1.0; do
    p_rate_percent=$(echo "$p_rate * 100" | bc);
    echo "Running with penetration rate ${p_rate_percent}%...";
    python "${SCRIPT_DIR}"/train_sac.py \
        --gpu -pr "${p_rate}" \
        --exp-name "vsl_pr_${p_rate_percent}" \
        > "logs/vsl_pr_${p_rate_percent}.log" 2>&1 &
done
wait;
echo "All done!"
