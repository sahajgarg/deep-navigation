# Regularizing Deep State Estimation with Kalman Filters
This project demonstrates that Kalman Filters can be effectively used to regularize predicitons from more expressive models 
like LSTMs when applied to the hidden states of LSTMs. The code includes a reimplementation of the paper "Backprop KF" with extensions.
Please read our paper describing the work in more detail here:

## Dependencies
Known dependencies include python3, pytorch, opencv. 

## Running the code
First, generate synthetic sequences, running `python generate_synthetic_sequences.py`. Then, run and save the model using `python main.py --save-model`. 
Play around with the arguments to get different, cool results!
