# Revolutionizing Student Dropout Prediction with Multi-Objective Reinforcement Learning: Balancing Accuracy and Earliness

## Abstract

Student Dropout Prediction (SDP) has emerged as a vital area in educational research, notably for its role in enhancing the understanding and prediction of student learning behaviors and trends within time series classification models. Traditional SDP approaches have primarily concentrated on either optimizing prediction accuracy or earliness, often failing to achieve a Pareto efficient/optimal solution that adequately balances these two objectives. This shortfall underscores the need for methodologies that adeptly manage the **trade-offs** between accuracy and timeliness in predictions.  Despite the common practice in scalarized RL methods of employing preset hyperparameters to transform multiple objectives into a singular goal, this approach can restrict the ability to generalize across the entire Pareto Frontier, potentially leading to compromises in either prediction accuracy or earliness. To address these challenges, our study introduces a novel **Multi-Objective Reinforcement Learning** (MORL) strategy, which redefines SDP as a **multi-objective optimization** problem. Our approach, diverging from traditional scalarized RL methods, employs a **vector reward** mechanism within a Multi-Objective Markov Decision Process (MOMDP) framework. This approach allows for a balanced consideration of both accuracy and earliness without the constraints of preset hyperparameters. By utilizing Double Deep Q-Network (DDQN) and **envelope Q-function** updates, our MORL model learns optimal policies across a spectrum of preferences, offering a more dynamic and adaptable solution to the multi-objective dilemma in SDP. The efficacy of our model has been rigorously validated through comprehensive evaluations on real-world MOOC datasets. These evaluations have demonstrated our model's superiority, outperforming existing methods in both predictive accuracy and earliness, thus marking a significant advancement in the field of SDP.

## Instructions

The experiments on two real-world MOOC datasets, **KDDCup2015** and **XuetangX**.

### `KDDCup2015`

* Example :  
`python train.py --env-name mooc --method crl-envelope --model cnn --gamma  0.99 --mem-size 500 --batch-size 256 --lr  1e-3 --epsilon 0.5 --weight-num 32 --episode-num 500 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.5 --name KDDCup2015`

The code for our envelope MOQ-learning algorithm is in `synthetic/crl/envelope/meta.py`, neural network architecture is configurable in `synthetic/crl/envelope/models`. 

### `XuetangX`

`python train.py --env-name mooc --method crl-envelope --model cnn --gamma  0.99 --mem-size 500 --batch-size 256 --lr  1e-3 --epsilon 0.5 --weight-num 32 --episode-num 500 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.5 --name XuetangX`

