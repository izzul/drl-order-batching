<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Deep reinforcement learning-based solution for a multi-objective online order batching problem</h3>

  <p align="center">
    Martijn Beeks, Reza Refaei Afshar, Yingqian Zhang, Remco Dijkman, Claudy van Dorst, Stijn de Looijer
    <br />
  </p>
</div>



<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains code for the presented DRL approach in the paper "Deep reinforcement learning-based solution for a multi-objective online order batching problem".



<!-- GETTING STARTED -->
## Getting Started

This is an example on how to setup this project locally on python 3.7.10.


1. Clone the repo
   ```sh
   git clone https://github.com/izzul/drl-order-batching.git
   ```
2. Install pip packages
   ```sh
   pip install -r requirements.txt
   ```
3. Install Hyperspace [Optional]
    ```sh
   cd drl-order-batching
   git clone https://github.com/yngtdd/hyperspace
   cd hyperspace
   python -m pip install .
   ```


<!-- USAGE EXAMPLES -->
## Usage

Similar to the paper, this repository consists out of several approaches.
1. Simulation model to test DRL approach
2. DRL approach
3. Hyper-parameter optimization of DRL approach

####2. DRL approach
A DRL approach for the warehousing problem has been trained with a stable-baselines library. This library requires a gym
environment for the models. The script drl_train.py defines a training job by setting the number of steps to take,
hyper-parameters of the learning model and etc. The scenario for this training job can be defined in drl_env.py where 
a yaml file with a scenario is loaded. When the training job is done, the model is saved at a predefined location.
To test a trained DRL approach, use drl_test.py and define the location of a trained model. This script will output the 
performance of a trained model in the console.
   ```sh
   python -m drl_train.py
   # not yet implemented
   # python -m drl_test.py
   ```
Config can be adjusted in config/scenario_tikno_1.yml in simulation section such as:

1. batching_opt
2. routing_opt
3. picker_number
4. cart_capacity

drl_train.py will only use the first orderFile to train the model