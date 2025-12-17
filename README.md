# STA analysis of a simple cell linear-nonlinear-poisson model
This code is from an assignment for a computational neuroscience course at the University of Rochester.

The purpose is to use a 2D Gabor function to implement a simple cell with a vertical RF, and an output nonlinearity of f(x) = x<sup>2</sup> if x > 0, otherwise f(x) = 0. The RF extended ±5 degrees in the x and y-directions. The image was discretized such that 1 pixel corresponds to a 0.2 x 0.2 degree span (50 x 50 image).
<img width="924" height="150" alt="2d_gabor" src="https://github.com/user-attachments/assets/77301a61-7145-4eb6-b099-8c7e5345f28c" />
#

The RF as a 2D image is:  
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/19daa421-43d7-4390-937a-fe34508da950" />

A cross-section of the RF at y = 0 along the x-axis.  
<img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/10cc52a6-481d-4188-9b3a-3dce93244756" />
#

Vertical and horizontal gratings were then created with the equation I(x,y,θ) = sin(k[x*(θ) - y*sin(θ)]), where θ is the orientation and k is the spatial frequency of 1/k = 0.56 degrees.  
<img width="320" height="240" alt="Figure_3" src="https://github.com/user-attachments/assets/80fc75cf-f5bf-422a-abe6-7894f47e2c5e" />
<img width="320" height="240" alt="Figure_5" src="https://github.com/user-attachments/assets/392e3ded-dfa5-44ad-ba2f-8b79f0076e98" />

These gratings were then presented to the simple cell and a spike rate was computed as a function of phase grating α for the vertical grating, and as a function of orientation θ for the horizontal grating.  
<img width="320" height="240" alt="Figure_4" src="https://github.com/user-attachments/assets/5dd138b2-765a-4814-bccb-a2c190175f07" />
<img width="320" height="240" alt="Figure_6" src="https://github.com/user-attachments/assets/4d1951f1-5100-4147-b71a-5c482a7e4b26" />

#
Lastly, reverse correlation was then performed on the simple cell model neurons via spike-triggered averaging. White noise images were created drawn from a Gaussian distribution N(0,1) and spiking responses were simulated assuming a Poisson model. The model was scaled such that the average spike count per image was 0.2 spikes.  
<img width="1000" height="400" alt="Figure_7" src="https://github.com/user-attachments/assets/0692043e-dc3f-43b0-acc8-9cf881eef269" />
<img width="1370" height="400" alt="Figure_8" src="https://github.com/user-attachments/assets/bd00ab78-b0f7-4261-8a40-c49807342d9c" />

As the number of white noise images presented to the simple cell model neuron increased, the correlation to the true RF increased linearly on a log scale towards 1.0.   
<img width="640" height="480" alt="Figure_9" src="https://github.com/user-attachments/assets/6fdb8012-71a9-4d90-8fc1-41c5c0c6b7d0" />
