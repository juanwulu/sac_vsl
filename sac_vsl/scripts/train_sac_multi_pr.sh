SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd );
echo "Running SAC on I80 VSL with multiple penetration rate settings...";
python ${SCRIPT_DIR}/train_sac.py --gpu -pr 0.1 --exp-name vsl_pr_10 > logs/vsl_pr_10.log 2>&1 &
python ${SCRIPT_DIR}/train_sac.py --gpu -pr 0.2 --exp-name vsl_pr_20 > logs/vsl_pr_20.log 2>&1 &
python ${SCRIPT_DIR}/train_sac.py --gpu -pr 0.5 --exp-name vsl_pr_50 > logs/vsl_pr_50.log 2>&1 &
python ${SCRIPT_DIR}/train_sac.py --gpu -pr 1.0 --exp-name vsl_pr_100 > logs/vsl_pr_100.log 2>&1 &
wait;
echo "All done!"