# PixelHop++

This project is an unofficial implementation of the paper [*PixelHop++: A Small Successive-Subspace-Learning-Based (SSL-based) Model for Image Classification*](https://arxiv.org/abs/2002.03141), a feed-forward image classification architecture based on purely traditional machine learning methods.

## Related works
This work is similar to the previous paper [*SSL (Successive Subspace Learning)*](https://arxiv.org/abs/1909.08190), which is a new machine learning methodology containing 4 key components: successive near-to-far neighborhood expansion, unsupervised dimension reduction via subspace approximation, supervised dimension reduction via label-assisted regression (LAG), and feature concatenation and decision making. The only differece here is it introduces the tree decomposition to make it more memory efficient.
For more intuition and mathematical explanation about the Saab transformation, read [*Interpretable Convolutional Neural Networks via Feedforward Design*](https://arxiv.org/abs/1810.02786).

 
## Experiment results
Parameter setting of module 1
![image](https://raw.githubusercontent.com/xshuai1996/PixelHopPlusPlus/master/results/params1.png)

Parameter setting of module 2
![image](https://raw.githubusercontent.com/xshuai1996/PixelHopPlusPlus/master/results/params2.png)

Parameter setting of module 3
![image](https://raw.githubusercontent.com/xshuai1996/PixelHopPlusPlus/master/results/params3.png)

Number of parameters in module 1
![image](https://raw.githubusercontent.com/xshuai1996/PixelHopPlusPlus/master/results/num_params1.png)

Number of parameters in module 2
![image](https://raw.githubusercontent.com/xshuai1996/PixelHopPlusPlus/master/results/num_params2.png)

Running time
![image](https://raw.githubusercontent.com/xshuai1996/PixelHopPlusPlus/master/results/time.png)

Accuracy
![image](https://raw.githubusercontent.com/xshuai1996/PixelHopPlusPlus/master/results/acc1.png)
![image](https://raw.githubusercontent.com/xshuai1996/PixelHopPlusPlus/master/results/acc2.png)

Confusion matrix acrossing different classes
![image](https://raw.githubusercontent.com/xshuai1996/PixelHopPlusPlus/master/results/acc3.png)

## To improve
Check out *improve.pdf*.