# FrictionLearner
Learning the friction of a non-trivial damping pendulum with hinge friction using Neural Networks (insight from Universal Differential Equations (UDE))

Simple pendulum has studied for long period of time and in real world experiments. There are non-conservative friction forces associated with the equation of motion. We use the new approach explained in [1] by using neural networks in the form of Universal Differential Equations (UDEs) to learn the complicated friction force. We use OPEN CV to extract the data from video frames of a real pendulum experiment and use the data for the training process.

[1] Rackauckas et. al, Univeral Differential Equatins for Scientific Machine Learning. (https://arxiv.org/abs/2001.04385)

 ### Learned friction(F(&theta;, &omega;)) as a function of time(t) using the simulated data:
 
 <p align="center">
  <img src="friction_vs_time_s_2InL1.png" width="700" height="430" />
 </p>

 ### Learned friction(F(&theta;, &omega;)) as a function of time(t) using real data from the pendulum experiment:
 
 <p align="center">
  <img src="ann64_Loss_theta_fric_05.png" width="700" height="430" />
 </p>
 
