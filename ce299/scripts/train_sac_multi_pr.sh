SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd );
echo "Running SAC on I80 VSL with multiple penetration rate settings...";
python ${SCRIPT_DIR}/train_sac.py -pr 0.05 --exp-name vsl_pr_0.05 > logs/vsl_pr_0.05.log 2>&1 &
python ${SCRIPT_DIR}/train_sac.py -pr 0.1 --exp-name vsl_pr_0.1 > logs/vsl_pr_0.1.log 2>&1 &
python ${SCRIPT_DIR}/train_sac.py -pr 0.2 --exp-name vsl_pr_0.2 > logs/vsl_pr_0.2.log 2>&1 &
wait;
echo "All done!"