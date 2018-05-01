# Optimization for Deep Learning

- [General](#general)  
- [Adaptive Gradient Methods](#adaptive-gradient-methods)
- [Batch Size](#batch-size)
- [Distributed Optimization](#distributed-optimization)  
- [Initialization](#initialization)  
- [Generalization](#generalization)
- [Loss Surface](#loss-surface)
- [Low Precision](#low-precision)
- [Normalization](#normalization)
- [Regularization](#regularization)
- [Meta Learning](#meta-learning)

## General
- 2016 ICML [Train faster, generalize better: Stability of stochastic gradient descent](http://proceedings.mlr.press/v48/hardt16.pdf)  
- 2016 arXiv [Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838)  
- 2016 Blog [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/index.html)  
- 2015 DL Summer School [Non-Smooth, Non-Finite, and Non-Convex Optimization](http://www.iro.umontreal.ca/~memisevr/dlss2015/2015_DLSS_NonSmoothNonFiniteNonConvex.pdf)  
- 2015 NIPS [Training Very Deep Networks](http://papers.nips.cc/paper/5850-training-very-deep-networks.pdf)  
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
- 2017 arXiv [Scaling SGD Batch Size to 32K for ImageNet Training](https://arxiv.org/pdf/1708.03888.pdf)  
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


## Generalization
- 2018 arXiv [On Characterizing the Capacity of Neural Networks using Algebraic Topology](https://arxiv.org/pdf/1802.04443.pdf)  
- 2017 arXiv [Exploring Generalization in Deep Learning](https://arxiv.org/pdf/1706.08947.pdf)  
- 2017 NIPS [Train longer, generalize better: closing the generalization gap in large batch training of neural networks](http://papers.nips.cc/paper/6770-train-longer-generalize-better-closing-the-generalization-gap-in-large-batch-training-of-neural-networks.pdf)  
- 2017 ICML [A Closer Look at Memorization in Deep Networks](https://arxiv.org/pdf/1706.05394.pdf)
- 2017 ICLR [Understanding deep learning requires rethinking generalization](https://openreview.net/pdf?id=Sy8gdB9xx)  

## Loss Surface
- 2017 arXiv [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/pdf/1712.09913.pdf)  
- 2017 arXiv [The loss surface and expressivity of deep convolutional neural networks](https://arxiv.org/pdf/1710.10928.pdf)  
- 2017 ICML [The Loss Surface of Deep and Wide Neural Networks](https://arxiv.org/pdf/1704.08045.pdf)  
- 2017 ICML [Geometry of Neural Network Loss Surfaces via Random Matrix Theory](http://proceedings.mlr.press/v70/pennington17a/pennington17a.pdf)  
- 2017 ICML [Sharp Minima Can Generalize For Deep Nets](https://arxiv.org/pdf/1703.04933.pdf)    
- 2017 ICLR [Entropy-SGD: Biasing Gradient Descent Into Wide Valleys](https://arxiv.org/pdf/1611.01838.pdf)  
- 2017 ICLR [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://openreview.net/pdf?id=H1oyRlYgg)  
- 2017 arXiv [An empirical analysis of the optimization of deep network loss surfaces](https://arxiv.org/pdf/1612.04010.pdf)  
- 2016 ICMLW [Visualizing Deep Network Training Trajectories with PCA](https://icmlviz.github.io/icmlviz2016/assets/papers/24.pdf)  
- 2016 ICLRW [Stuck in a What? Adventures in Weight Space](https://arxiv.org/pdf/1602.07320.pdf)  
- 2015 ICLR [Qualitatively Characterizing Neural Network Optimization Problems](https://arxiv.org/pdf/1412.6544.pdf)  
- 2015 AISTATS [The Loss Surfaces of Multilayer Networks](http://www.jmlr.org/proceedings/papers/v38/choromanska15.pdf)  
- 2014 NIPS [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](http://papers.nips.cc/paper/5486-identifying-and-attacking-the-saddle-point-problem-in-high-dimensional-non-convex-optimization.pdf)  

## Low Precision
- 2017 arXiv [Gradient Descent for Spiking Neural Networks](https://arxiv.org/abs/1706.04698)  
- 2017 arXiv [Training Quantized Nets: A Deeper Understanding](https://arxiv.org/abs/1706.02379)  
- 2017 arXiv [TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning](https://arxiv.org/abs/1705.07878)  
- 2017 ICML [ZipML: Training Linear Models with End-to-End Low Precision](http://proceedings.mlr.press/v70/zhang17e/zhang17e.pdf)  
- 2016 arXiv [QSGD: Communication-Optimal Stochastic Gradient Descent, with Applications to Training Neural Networks](https://arxiv.org/pdf/1610.02132.pdf)  
- 2015 NIPS [Taming the Wild: A Unified Analysis of Hogwild!-Style Algorithms](https://pdfs.semanticscholar.org/a1d2/1f6c8eef605bf132179daf717a232774b375.pdf)  
- 2013 arXiv [Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation](https://arxiv.org/pdf/1308.3432.pdf)  

## Noise
- 2015 arXiv [Adding Gradient Noise Improves Learning for Very Deep Networks](http://arxiv.org/abs/1511.06807)      

## Normalization
- 2017 arXiv [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)  
- 2017 arXiv [Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models](https://arxiv.org/abs/1702.03275)  
- 2016 NIPS [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/pdf/1602.07868.pdf)  
- 2016 NIPS [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)  
- 2016 ICML [Normalization Propagation: A Parametric Technique for Removing Internal Covariate Shift in Deep Networks](https://arxiv.org/pdf/1603.01431.pdf)    
- 2016 ICLR [Data-Dependent Path Normalization in Neural Networks](http://arxiv.org/pdf/1511.06747v4.pdf)  
- 2015 NIPS [Path-SGD: Path-Normalized Optimization in Deep Neural Networks](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2015_5797.pdf)  
- 2015 ICML [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://jmlr.org/proceedings/papers/v37/ioffe15.pdf)  

## Regularization  
- 2017 arXiv [L2 Regularization versus Batch and Weight Normalization](https://arxiv.org/abs/1706.05350)  
- 2014 JMLR [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) (Dropout)   

## Meta-Learning  
- 2017 ICML [Neural Optimizer Search with Reinforcement Learning](https://arxiv.org/pdf/1709.07417.pdf)  
- 2017 ICML [Learned Optimizers that Scale and Generalize](https://arxiv.org/pdf/1703.04813.pdf)  
- 2017 ICML [Learning to Learn without Gradient Descent by Gradient Descent](http://www.cantab.net/users/yutian.chen/Publications/ChenEtAl_ICML17_L2L.pdf)  
- 2017 ICLR [Learning to Optimize](https://openreview.net/pdf?id=ry4Vrt5gl)  
- 2016 arXiv [Learning to reinforcement learn](https://arxiv.org/abs/1611.05763)  
- 2016 NIPSW [Learning to Learn for Global Optimization of Black Box Functions](https://arxiv.org/abs/1611.03824)  
- 2016 NIPS [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474)    
- 2016 ICML [Meta-learning with memory-augmented neural networks](http://proceedings.mlr.press/v48/santoro16.pdf)  

## Hyperparameter
- 2015 ICML [Gradient-based hyperparameter optimization through reversible learning](https://www.robots.ox.ac.uk/~vgg/rg/papers/MaclaurinICML15.pdf)  

## Bayesian Optimization  
- 2012 [Practical Bayesian Optimization of Machine Learning Algorithms](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)  
