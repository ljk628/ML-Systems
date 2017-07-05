# Optimization for Deep Learning

- [General](#general)  
- [Adaptive Gradient Methods](#adaptive-gradient-methods)
- [Batch Size](#batch-size)
- [Distributed Optimization](#distributed-optimization)  
- [Initialization](#initialization)  
- [Learning Rate](#learning-rate)  
- [Loss Surface](#loss-surface)
- [Normalization](#normalization)
- [Regularization](#regularization)
- [Meta Learning](#meta-learning)

## General
- 2016 ICML [Train faster, generalize better: Stability of stochastic gradient descent](http://proceedings.mlr.press/v48/hardt16.pdf)  
- 2016 arXiv [Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838)  
- 2016 Blog [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/index.html)  
- 2015 DL Summer School [Non-Smooth, Non-Finite, and Non-Convex Optimization](http://www.iro.umontreal.ca/~memisevr/dlss2015/2015_DLSS_NonSmoothNonFiniteNonConvex.pdf)  
- 2015 NIPS [Training Very Deep Networks](http://papers.nips.cc/paper/5850-training-very-deep-networks.pdf)  
- 2015 ICLR [Qualitatively Characterizing Neural Network Optimization Problems](https://arxiv.org/pdf/1412.6544.pdf)  
- 2015 AISTATS [Deeply-Supervised Nets](http://jmlr.org/proceedings/papers/v38/lee15a.pdf)  
- 2014 OSLW [On the Computational Complexity of Deep Learning](http://lear.inrialpes.fr/workshop/osl2015/slides/osl2015_shalev_shwartz.pdf)  
- 2011 ICML [On Optimization Methods for Deep Learning](http://ai.stanford.edu/~quocle/LeNgiCoaLahProNg11.pdf)  
- 2010 AISTATS [Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)  

## Adaptive Gradient Methods
- 2017 [The Marginal Value of Adaptive Gradient Methods in Machine Learning](https://arxiv.org/abs/1705.08292)  
- 2017 ICLR [SGDR: Stochastic Gradient Descent with Restarts](https://openreview.net/pdf?id=Skq89Scxx)  
- 2015 ICLR [Adam: A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980) (Adam)  
- 2013 ICML [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf) (NAG)  
- 2012 Lecture [RMSProp: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning]() (RMSProp)  
- 2011 JMLR [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) (Adagrad)  

## Batch Size
- 2017 arXiv [Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)  
- 2017 ICML [Sharp Minima Can Generalize For Deep Nets](https://arxiv.org/abs/1703.04933)  
- 2017 ICLR [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://openreview.net/pdf?id=H1oyRlYgg)  

## Distributed Optimization  
- 2017 arXiv [Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)  
- 2017 arXiv [TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning](https://arxiv.org/pdf/1705.07878.pdf)  
- 2017 arXiv [QSGD: Communication-Efficient Stochastic Gradient Descent, with Applications to Training Neural Networks](https://arxiv.org/pdf/1610.02132.pdf) (QSGD)  
- 2016 ICML [Training Neural Networks Without Gradients: A Scalable ADMM Approach](http://jmlr.org/proceedings/papers/v48/taylor16.pdf)  
- 2016 IJCAI [Staleness-aware Async-SGD for Distributed Deep Learning](http://www.ijcai.org/Proceedings/16/Papers/335.pdf)  
- 2016 ICLRW [Revisiting Distributed Synchronous SGD](http://arxiv.org/abs/1604.00981)  
- 2016 Thesis [Distributed Stochastic Optimization for Deep Learning](https://cs.nyu.edu/media/publications/zhang_sixin.pdf) (EASGD)    
- 2015 NIPS [Deep learning with Elastic Averaging SGD](https://www.cs.nyu.edu/~zsx/nips2015.pdf) (EASGD)  
- 2015 ICLR [Parallel training of Deep Neural Networks with Natural Gradient and Parameter Averaging](http://arxiv.org/pdf/1409.1556v6.pdf)  

## Initialization
- 2016 NIPS [Toward Deeper Understanding of Neural Networks: The Power of Initialization and a Dual View on Expressivity](http://papers.nips.cc/paper/6427-toward-deeper-understanding-of-neural-networks-the-power-of-initialization-and-a-dual-view-on-expressivity.pdf)
- 2016 ICLR [All You Need is a Good Init](https://arxiv.org/pdf/1511.06422.pdf)  
- 2016 ICLR [Data-dependent Initializations of Convolutional Neural Networks](https://arxiv.org/pdf/1511.06856.pdf)    
- 2015 ICCV [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://research.microsoft.com/en-us/um/people/kahe/publications/iccv15imgnet.pdf) (MSRAinit)   
- 2014 ICLR [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/pdf/1312.6120.pdf)  
- 2013 ICML [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)  
- 2010 AISTATS [Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) (Xavier initialization)  

## Loss Surface
- 2017 ICML [Sharp Minima Can Generalize For Deep Nets](https://arxiv.org/pdf/1703.04933.pdf)    
- 2017 ICLR [Entropy-SGD: Biasing Gradient Descent Into Wide Valleys](https://arxiv.org/pdf/1611.01838.pdf)  
- 2017 ICLR [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://openreview.net/pdf?id=H1oyRlYgg)  
- 2015 AISTATS [The Loss Surfaces of Multilayer Networks](http://www.jmlr.org/proceedings/papers/v38/choromanska15.pdf)  
- 2014 NIPS [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](http://papers.nips.cc/paper/5486-identifying-and-attacking-the-saddle-point-problem-in-high-dimensional-non-convex-optimization.pdf)  

## Noise
- 2015 arXiv [Adding Gradient Noise Improves Learning for Very Deep Networks](http://arxiv.org/abs/1511.06807)      

## Normalization
- 2017 arXiv [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)  
- 2017 arXiv [Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models](https://arxiv.org/abs/1702.03275)  
- 2016 NIPS [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/pdf/1602.07868.pdf)  
- 2016 NIPS [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)  
- 2016 ICLR [Data-Dependent Path Normalization in Neural Networks](http://arxiv.org/pdf/1511.06747v4.pdf)  
- 2015 NIPS [Path-SGD: Path-Normalized Optimization in Deep Neural Networks](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2015_5797.pdf)  
- 2015 ICML [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://jmlr.org/proceedings/papers/v37/ioffe15.pdf)  

## Regularization  
- 2017 arXiv [L2 Regularization versus Batch and Weight Normalization](https://arxiv.org/abs/1706.05350)  
- 2014 JMLR [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) (Dropout)   

## Meta-Learning  
- 2017 ICLR [Learning to Optimize](https://openreview.net/pdf?id=ry4Vrt5gl)  
- 2016 arXiv [Learning to reinforcement learn](https://arxiv.org/abs/1611.05763)  
- 2016 NIPSW [Learning to Learn for Global Optimization of Black Box Functions](https://arxiv.org/abs/1611.03824)  
- 2016 NIPS [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474)    

## Bayesian Optimization  
- 2012 [Practical Bayesian Optimization of Machine Learning Algorithms](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)  

## Stochastic Neurons
- 2013 arXiv [Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation](https://arxiv.org/pdf/1308.3432.pdf)  
