# Implicit Quantile Networks (IQN) for Distributional Reinforcement Learning and Extensions
PyTorch Implementation of Implicit Quantile Networks (IQN) for Distributional Reinforcement Learning with additional extensions like PER, Noisy layer and N-step bootstrapping. Creating a new Rainbow-DQN version. 
This implementation allows it also to run and train on several environments in parallel!



### Implementation
- IQN with all extensions: [run.py](https://github.com/BY571/IQN/blob/master/run.py)
The IQN implementation in this repository is already a Double IQN version with target networks! 

### Extensions

- Dueling IQN
- Noisy layer
- N-step bootstrapping 
- [Munchausen](https://medium.com/analytics-vidhya/munchausen-reinforcement-learning-9876efc829de) RL 
- Parallel environments for faster training (wall clock time). For CartPole-v1 3 worker reduced training time to 1/3! 

## Train
It is possible to train on simple environments like CartPole-v1 and LunarLander-v2 or on Atari games with image inputs!

### Quick Start - CartPole-v1
```bash
# Activate virtual environment first
source venv/bin/activate

# Run with optimized hyperparameters for CartPole
python run.py -env CartPole-v1 -info iqn_cartpole -frames 20000 -eval_every 5000 -N 8 -lr 2.5e-4 -bs 32 -eps_frames 5000 -w 1
```

**Important for CartPole:** The default epsilon decay (`eps_frames=1000000`) is too slow for CartPole. Use `-eps_frames 2000-5000` for proper learning!

### Atari Games
To run on the Atari game Pong:
```bash
python run.py -env PongNoFrameskip-v4 -info iqn_pong1
```

### Advanced CartPole (Faster Learning)
```bash
python run.py -env CartPole-v1 -info iqn_fast -frames 20000 -eval_every 5000 -N 8 -lr 5e-4 -bs 64 -eps_frames 2000 -w 1
```

#### Other hyperparameter and possible inputs
To see the options:
`python run.py -h`

    -agent, choices=["iqn","iqn+per","noisy_iqn","noisy_iqn+per","dueling","dueling+per", "noisy_dueling","noisy_dueling+per"], Specify which type of IQN agent you want to train, default is IQN - baseline!
    -env,  Name of the Environment, default = BreakoutNoFrameskip-v4
    -frames, Number of frames to train, default = 10 mio
    -eval_every, Evaluate every x frames, default = 250000
    -eval_runs, Number of evaluation runs, default = 2
    -seed, Random seed to replicate training runs, default = 1
    -munchausen, choices=[0,1], Use Munchausen RL loss for training if set to 1 (True), default = 0
    -bs, --batch_size, Batch size for updating the DQN, default = 8
    -layer_size, Size of the hidden layer, default=512
    -n_step, Multistep IQN, default = 1
    -N, Number of quantiles, default = 8
    -m, --memory_size, Replay memory size, default = 1e5
    -lr, Learning rate, default = 2.5e-4
    -g, --gamma, Discount factor gamma, default = 0.99
    -t, --tau, Soft update parameter tat, default = 1e-3
    -eps_frames, Linear annealed frames for Epsilon, default = 1 mio
    -min_eps, Final epsilon greedy value, default = 0.01
    -info, Name of the training run
    -w, --worker, Number of parallel environments. Batch size increases proportional to number of worker. Not recommended to have more than 4 worker, default = 1
    -save_model, choices=[0,1]  Specify if the trained network shall be saved or not, default is 0 - not saved!

### Observe training results
  `tensorboard --logdir=runs`
  
### Remote TensorBoard Viewing

If you're training on a remote server and want to view TensorBoard on your local machine:

1. **On the remote server**, start TensorBoard:
   ```bash
   source venv/bin/activate
   python -m tensorboard.main --logdir=runs --host=127.0.0.1 --port=6006 --load_fast=false
   ```

2. **On your local machine**, create an SSH tunnel:
   ```bash
   ssh -L 6006:localhost:6006 username@remote-server-ip
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:6006
   ```

**Note:** If port 6006 is already in use on your local machine, you can use a different port:
- SSH tunnel: `ssh -L 8888:localhost:6006 username@remote-server-ip`
- Browser: `http://localhost:8888`

#### Dependencies
<pre>
Python 3.10+ 
PyTorch 2.0+  
Numpy 1.20+ 
gymnasium 0.29+ 
tensorboard
opencv-python (for Atari environments)
</pre>

## CartPole Results
IQN and Extensions (default hyperparameter):
![alttext](/imgs/IQN_CP_.png)

Dueling IQN and Extensions (default hyperparameter):
![alttext](/imgs/Dueling_IQN_CP_.png)


## Atari Results
IQN and M-IQN comparison (only trained for 500000 frames ~ 140 min).


**Hyperparameter:**
- frames 500000
- eps_frames 75000
- min_eps 0.025
- eval_every 10000 
- lr 1e-4 
- t 5e-3 
- m 15000 
- N 32

![alttext](/imgs/IQN_MIQN_BREAKOUT_.png)

Performance after 10 mio frames, score 258 

![](/imgs/Breakout_IQN.gif?)

## ToDo:
- Comparison plot for n-step bootstrapping (n-step bootstrapping with n=3 seems to give a strong boost in learning compared to one step bootstrapping, plots will follow) 
- Performance plot for Pong compared with Rainbow
- adding [Munchausen](https://medium.com/analytics-vidhya/munchausen-reinforcement-learning-9876efc829de) RL &#x2611;


## Help and issues:
Im open for feedback, found bugs, improvements or anything. Just leave me a message or contact me.

### Paper references:

- [IQN](https://arxiv.org/abs/1806.06923)
- [Dueling DQN](https://arxiv.org/abs/1511.06581)
- [Noisy layer](https://arxiv.org/pdf/1706.10295.pdf)
- [C51](https://arxiv.org/pdf/1707.06887.pdf)
- [PER](https://arxiv.org/pdf/1511.05952.pdf)


## Author
- Sebastian Dittert

**Feel free to use this code for your own projects or research.**
For citation:
```
@misc{IQN and Extensions,
  author = {Dittert, Sebastian},
  title = {Implicit Quantile Networks (IQN) for Distributional Reinforcement Learning and Extensions},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/BY571/IQN}},
}
