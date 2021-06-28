CONFIG_PATH=configs/ddl05_doom.config
for i in `seq 1 64`;
do
  echo worker $i
  CUDA_VISIBLE_DEVICES=-1 python extract.py -c $CONFIG_PATH &
  sleep 1.0
done
# can't just use 'wait' b/c doom env leaves processes open
N_RUNNING=100 # some number > 0
while [$N_RUNNING > 0]
do
  N_RUNNING=$(ps aux | grep -c extract)
  sleep 60.0 # wait a minute
done
pkill -9 -f vizdoom # clean up left over doom env processes
CUDA_VISIBLE_DEVICES=0 python vae_train.py -c $CONFIG_PATH 
CUDA_VISIBLE_DEVICES=0 python series.py -c $CONFIG_PATH
CUDA_VISIBLE_DEVICES=0 python rnn_train.py -c $CONFIG_PATH 
CUDA_VISIBLE_DEVICES=-1 python train.py -c $CONFIG_PATH
