# **Reinforcement Learning Algorithm Comparison**

A comprehensive comparison of four reinforcement learning algorithms (DQN, REINFORCE, A2C, PPO) on the CartPole-v1 environment.

##  **Overview**

This project implements and compares four popular reinforcement learning algorithms:

* **Deep Q-Network (DQN)** \- Value-based off-policy method  
* **REINFORCE** \- Vanilla policy gradient method  
* **Advantage Actor-Critic (A2C)** \- On-policy actor-critic method  
* **Proximal Policy Optimization (PPO)** \- Advanced policy gradient method

Each algorithm is tested with 10 different hyperparameter configurations (40 total configurations) to identify optimal settings and understand algorithm behavior.

**Environment**: Custom env

## **Algorithms Implemented**

### **1\. Deep Q-Network (DQN)**

* Experience replay buffer  
* Target network for stability  
* Epsilon-greedy exploration  
* Double Q-learning (optional)

### **2\. REINFORCE**

* Monte Carlo policy gradient  
* Baseline subtraction for variance reduction  
* Entropy regularization  
* Full trajectory learning

### **3\. Advantage Actor-Critic (A2C)**

* Synchronous actor-critic updates  
* N-step returns  
* Generalized Advantage Estimation (GAE)  
* Shared network architecture

### **4\. Proximal Policy Optimization (PPO)**

* Clipped surrogate objective  
* Multiple epochs per batch  
* Value function clipping  
* Adaptive KL penalty (optional)

##  **Requirements**

python \>= 3.8  
torch \>= 1.9.0  
gymnasium \>= 0.26.0 (or gym \>= 0.21.0)  
numpy \>= 1.21.0  
matplotlib \>= 3.4.0

pandas \>= 1.3.0

##  **Installation**

1. Clone the repository:

bash  
https://github.com/AkotoChristine/AkotoNimohChristine\_rl\_summative

cd rl-algorithm-comparison

2. Create a virtual environment:

bash  
python \-m venv venv

source venv/bin/activate  *\# On Windows: venv\\Scripts\\activate*

3. Install dependencies:

bash

pip install \-r requirements.txt

##  **Usage**

### **Training All Configurations**

bash

Train each file under training/

This will train all 40 configurations (10 per algorithm) and save results.

### **Training Individual Algorithms**

bash  
*\# Train DQN*  
python train\_dqn.py \--config configs/dqn\_baseline.json

*\# Train REINFORCE*  
python train\_reinforce.py \--config configs/reinforce\_baseline.json

*\# Train A2C*  
python train\_a2c.py \--config configs/a2c\_baseline.json

*\# Train PPO*

python train\_ppo.py \--config configs/ppo\_baseline.json

### **Evaluating Trained Models**

bash

python evaluate.py \--model models/a2c\_lowgamma.pth \--episodes 100

### **Generating Visualizations**

bash

python visualize\_results.py \--results results.pkl

## ** Project Structure**

rl-algorithm-comparison/  
├── algorithms/  
│   ├── dqn.py              \# DQN implementation  
│   ├── reinforce.py        \# REINFORCE implementation  
│   ├── a2c.py              \# A2C implementation  
│   └── ppo.py              \# PPO implementation  
├── configs/  
│   ├── dqn\_configs.json    \# DQN hyperparameter configs  
│   ├── reinforce\_configs.json  
│   ├── a2c\_configs.json  
│   └── ppo\_configs.json  
├── models/                 \# Saved model checkpoints  
│   ├── dqn/  
│   ├── reinforce/  
│   ├── a2c/  
│   └── ppo/  
├── results/  
│   ├── training\_logs/      \# Training logs and metrics  
│   ├── plots/              \# Generated visualizations  
│   └── results.pkl         \# Compiled results  
├── utils/  
│   ├── replay\_buffer.py    \# Experience replay buffer  
│   ├── network.py          \# Neural network architectures  
│   └── logger.py           \# Logging utilities  
├── train\_all.py            \# Train all configurations  
├── train\_dqn.py            \# Train DQN  
├── train\_reinforce.py      \# Train REINFORCE  
├── train\_a2c.py            \# Train A2C  
├── train\_ppo.py            \# Train PPO  
├── evaluate.py             \# Evaluate trained models  
├── visualize\_results.py    \# Generate plots  
├── requirements.txt        \# Python dependencies

└── README.md               \# This file

##  **Results Summary**

### **Overall Performance Rankings**

| Rank | Algorithm | Configuration | Mean Reward | Std Reward |
| ----- | ----- | ----- | ----- | ----- |
| 1 | A2C | LowGamma | 483.18 | 335.45 |
| 2 | DQN | A2C\_Baseline | 354.24 | 233.36 |
| 3 | DQN | A2C\_HighLR | 286.81 | 245.76 |
| 4 | A2C | AggressiveExplore | 274.97 | 220.11 |
| 5 | PPO | PPO\_HighEntropy | 262.97 | 276.63 |

### **Best Configuration Per Algorithm**

| Algorithm | Best Config | Mean Reward | Episodes to Converge |
| ----- | ----- | ----- | ----- |
| A2C | LowGamma | 483.18 | \~120 |
| DQN | A2C\_Baseline | 354.24 | \~180 |
| PPO | PPO\_HighEntropy | 262.97 | \~250 |
| REINFORCE | REINFORCE\_HighLR | 116.52 | \~400 |

### **Key Findings**

* **A2C** achieved the highest mean reward with proper hyperparameter tuning  
* **DQN** demonstrated the best stability and generalization  
* **PPO** showed robust performance across different hyperparameters  
* **REINFORCE** suffered from high variance and poor sample efficiency

##  **Hyperparameter Configurations**

### **DQN Configurations**

| Config Name | Learning Rate | Gamma | Buffer Size | Batch Size | Epsilon Decay |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Baseline | 0.001 | 0.99 | 10000 | 64 | 0.995 |
| HighLR | 0.005 | 0.99 | 10000 | 64 | 0.995 |
| LowLR | 0.0001 | 0.99 | 10000 | 64 | 0.995 |
| LargeBuffer | 0.001 | 0.99 | 50000 | 64 | 0.995 |
| SmallBatch | 0.001 | 0.99 | 10000 | 32 | 0.995 |
| LargeBatch | 0.001 | 0.99 | 10000 | 128 | 0.995 |
| HighGamma | 0.001 | 0.999 | 10000 | 64 | 0.995 |
| LowGamma | 0.001 | 0.95 | 10000 | 64 | 0.995 |
| AggressiveExplore | 0.001 | 0.99 | 10000 | 64 | 0.990 |
| ConservativeExplore | 0.001 | 0.99 | 10000 | 64 | 0.999 |

### **REINFORCE Configurations**

| Config Name | Learning Rate | Gamma | Hidden Size | Entropy Coef | Baseline |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Baseline | 0.001 | 0.99 | 128 | 0.0 | Yes |
| HighLR | 0.01 | 0.99 | 128 | 0.0 | Yes |
| LowLR | 0.0001 | 0.99 | 128 | 0.0 | Yes |
| HighGamma | 0.001 | 0.999 | 128 | 0.0 | Yes |
| LowGamma | 0.001 | 0.95 | 128 | 0.0 | Yes |
| LargeNet | 0.001 | 0.99 | 256 | 0.0 | Yes |
| SmallNet | 0.001 | 0.99 | 64 | 0.0 | Yes |
| HighEntropy | 0.001 | 0.99 | 128 | 0.05 | Yes |
| VeryHighEntropy | 0.001 | 0.99 | 128 | 0.1 | Yes |
| Balanced | 0.001 | 0.99 | 128 | 0.01 | Yes |

### **A2C Configurations**

| Config Name | Learning Rate | Gamma | N-Steps | Entropy Coef | VF Coef |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Baseline | 0.001 | 0.99 | 5 | 0.01 | 0.5 |
| HighLR | 0.005 | 0.99 | 5 | 0.01 | 0.5 |
| LowLR | 0.0001 | 0.99 | 5 | 0.01 | 0.5 |
| HighGamma | 0.001 | 0.999 | 5 | 0.01 | 0.5 |
| LowGamma | 0.001 | 0.95 | 5 | 0.01 | 0.5 |
| MoreSteps | 0.001 | 0.99 | 10 | 0.01 | 0.5 |
| FewerSteps | 0.001 | 0.99 | 2 | 0.01 | 0.5 |
| HighEntropy | 0.001 | 0.99 | 5 | 0.05 | 0.5 |
| AggressiveExplore | 0.001 | 0.99 | 5 | 0.05 | 0.5 |
| ConservativeExplore | 0.001 | 0.99 | 5 | 0.001 | 0.5 |

### **PPO Configurations**

| Config Name | Learning Rate | Gamma | Clip Range | Entropy Coef | Epochs |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Baseline | 0.0003 | 0.99 | 0.2 | 0.01 | 4 |
| HighLR | 0.001 | 0.99 | 0.2 | 0.01 | 4 |
| LowLR | 0.0001 | 0.99 | 0.2 | 0.01 | 4 |
| LargeClip | 0.0003 | 0.99 | 0.3 | 0.01 | 4 |
| SmallClip | 0.0003 | 0.99 | 0.1 | 0.01 | 4 |
| HighEntropy | 0.0003 | 0.99 | 0.2 | 0.05 | 4 |
| MoreEpochs | 0.0003 | 0.99 | 0.2 | 0.01 | 10 |
| FewerEpochs | 0.0003 | 0.99 | 0.2 | 0.01 | 2 |
| LargeBatch | 0.0003 | 0.99 | 0.2 | 0.01 | 4 |
| SmallBatch | 0.0003 | 0.99 | 0.2 | 0.01 | 4 |

## **Visualization**

The project generates several visualizations:

1. **Performance Comparison**: Bar chart comparing mean rewards across all configurations  
2. **Learning Curves**: Episode reward progression for each configuration  
3. **Convergence Analysis**: Episodes required to reach stable performance  
4. **Stability Analysis**: Training loss and policy entropy over time  
5. **Generalization Tests**: Performance on unseen initial states

Example visualizations are saved in `All_Visuals`.

##  **Reproducing Results**

To reproduce the exact results from the paper:

1. Set random seeds:

python  
torch.manual\_seed(42)  
np.random.seed(42)

random.seed(42)

2. Run training:

bash

python train\_all.py \--seed 42 \--episodes 500

3. Generate plots:

bash

python visualize\_results.py \--results results/results.pkl

##  **Known Issues**

* PyBullet warnings (`Remove body failed`) can be safely ignored  
* Pandas DataFrame creation may fail on some systems \- use manual table formatting  
* High variance in REINFORCE makes results sensitive to random seeds

##  **References**

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*.  
2. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*.  
3. Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. *ICML*.  
4. Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv*.

##  **Contributing**

Contributions are welcome\! Please:

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit your changes (`git commit -m 'Add amazing feature'`)  
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

##  **License**

This project is licensed under the MIT License \- see the LICENSE file for details.

##  **Authors**

* Your Name \- Akoto-Nimoh Christine   
* GitHub: [@](https://github.com/yourusername)AkotoChristine

##  **Acknowledgments**

* OpenAI Gym for the CartPole environment  
* PyTorch team for the deep learning framework  
* Spinning Up in Deep RL by OpenAI for algorithm implementations  
* Stable-Baselines3 for reference implementations

## **Contact**

For questions or feedback, please open an issue or contact c.akotonimo@alustudent.com

**Note**: This project was developed as part of a reinforcement learning course project. Results may vary depending on hardware, random seeds, and environment versions.
