source venv/bin/activate

python3 test_gpu.py
python3 download_data.py
python3 prepare_data.py

screen -mS train_text bash -lc 'cd /home/dominic/orion && source venv/bin/activate && python3 train_text.py'
screen -dmS tensorboard bash -lc 'cd /home/dominic/orion && source venv/bin/activate && tensorboard --logdir=constellation_one_text --host=0.0.0.0 --port=6007'