## Entity Detection

### How to use

run the script
```
bash get_data.sh
```
to get data.

```
python3.6 train.py --no_cuda
or
python3.6 train.py
```

to train the model. You should stop the training process manually when you want to terminate it.

```
python3.6 train.py --test
```

to test the model.

### Preprocessing

Training data can be generated by running script

```
python3.6 preprocess.py
```

### preliminary Result

Run 200 epoch.
DEV  85.200553%
TEST 83.695301%

### To Do

Output or Logging should be improved. 
