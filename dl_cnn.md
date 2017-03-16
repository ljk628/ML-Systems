# Convolutional Nerual Netowrks

- [ImageNet Models](#imagenet-models)  
- [Architecture Design](#architecture-design)
- [Visualization](#visualization)
- [Fast Convolution](#fast-convolution)
- [Low-Rank Filter Approximation](#low-rank-filter-approximation)
- [Low Precision](#low-precision)  
- [Parameter Pruning](#parameter-pruning)  
- [Transfer Learning](#transfer-learning)  

## ImageNet Models  
- 2016 ECCV [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (Pre-ResNet)   
- 2016 arXiv [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](http://arxiv.org/abs/1602.07261) (Inception V4)  
- 2015 arXiv [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385) (ResNet)     
- 2015 arXiv [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567) (Inception V3)  
- 2015 ICML [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://jmlr.org/proceedings/papers/v37/ioffe15.pdf) (Inception V2)  
- 2015 ICCV [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://research.microsoft.com/en-us/um/people/kahe/publications/iccv15imgnet.pdf) (PReLU)  
- 2015 ICLR [Very Deep Convolutional Networks For Large-scale Image Recognition](http://arxiv.org/abs/1409.1556) (VGG)  
- 2015 CVPR [Going Deeper with Convolutions](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf) (GoogleNet/Inception V1)   
- 2012 NIPS [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (AlexNet)  

## Architecture Design
- 2017 arXiv [Large-Scale Evolution of Image Classifiers](https://arxiv.org/abs/1703.01041)  
- 2017 ICLR [Neural Architecture Search with Reinforcement Learning](https://openreview.net/pdf?id=r1Ue8Hcxg)  
- 2017 ICLR [Designing Neural Network Architectures using Reinforcement Learning](https://openreview.net/pdf?id=S1c2cvqee)  
- 2016 NIPS [Residual Networks Behave Like Ensembles of Relatively Shallow Networks](https://arxiv.org/abs/1605.06431)  
- 2016 arXiv [Densely Connected Convolutional Networks](http://arxiv.org/abs/1608.06993)  
- 2016 BMVC [Wide Residual Networks](http://arxiv.org/abs/1605.07146)  
- 2016 arXiv [Do Deep Convolutional Nets Really Need to be Deep and Convolutional?](https://arxiv.org/abs/1603.05691)  
- 2016 arXiv [Benefits of depth in neural networks](http://arxiv.org/abs/1602.04485)  
- 2016 AAAI [On the Depth of Deep Neural Networks: A Theoretical View](http://arxiv.org/abs/1506.05232)  
- 2016 arXiv [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <1MB model size](http://arxiv.org/abs/1602.07360)  
- 2015 ICMLW [Highway Networks](http://arxiv.org/pdf/1505.00387v2.pdf)  
- 2015 CVPR [Convolutional Neural Networks at Constrained Time Cost](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/He_Convolutional_Neural_Networks_2015_CVPR_paper.pdf)   
- 2014 NIPS [Do Deep Nets Really Need to be Deep?](http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf)  
- 2014 ICLRW [Understanding Deep Architectures using a Recursive Convolutional Network](http://arxiv.org/abs/1312.1847)  
- 2009 ICCV [What is the Best Multi-Stage Architecture for Object Recognition?](http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf)   
- 1994 T-NN [SVD-NET: An Algorithm that Automatically Selects Network Structure](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?reload=true&arnumber=286929)  

## Visualization

- 2015 ICMLW [Understanding Neural Networks Through Deep Visualization](http://yosinski.com/media/papers/Yosinski__2015__ICML_DL__Understanding_Neural_Networks_Through_Deep_Visualization__.pdf)  
- 2014 ECCV [Visualizing and Understanding Convolutional Networks](https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)  

## Fast Convolution

- 2017 ICLR [Faster CNNs with Direct Sparse Convolutions and Guided Pruning](https://openreview.net/pdf?id=rJPcZ3txx)  
- 2016 NIPS [PerforatedCNNs: Acceleration through Elimination of Redundant Convolutions](http://arxiv.org/abs/1504.08362)  
- 2016 CVPR [Fast Algorithms for Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf) (Winograd)  
- 2015 CVPR [Sparse Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.pdf)  

## Low-Rank Filter Approximation
- 2016 ICLR [Convolutional Neural Networks with Low-rank Regularization](https://arxiv.org/abs/1511.06067)  
- 2016 ICLR [Training CNNs with Low-Rank Filters for Efficient Image Classification](http://arxiv.org/abs/1511.06744)  
- 2016 TPAMI [Accelerating Very Deep Convolutional Networks for Classification and Detection](https://arxiv.org/abs/1505.06798)  
- 2015 CVPR [Efficient and Accurate Approximations of Nonlinear Convolutional Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Efficient_and_Accurate_2015_CVPR_paper.pdf)  
- 2015 ICLR [Speeding-up convolutional neural networks using fine-tuned cp-decomposition](https://arxiv.org/pdf/1412.6553v3.pdf)  
- 2014 NIPS [Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation](http://papers.nips.cc/paper/5544-exploiting-linear-structure-within-convolutional-networks-for-efficient-evaluation.pdf)  
- 2014 BMVC [Speeding up Convolutional Neural Networks with Low Rank Expansions](https://arxiv.org/abs/1405.3866)  
- 2013 NIPS [Predicting Parameters in Deep Learning](https://papers.nips.cc/paper/5025-predicting-parameters-in-deep-learning.pdf)  
- 2013 CVPR [Learning Separable Filters](http://cvlabwww.epfl.ch/~lepetit/papers/rigamonti_cvpr13.pdf)  

## Low Precision
- 2017 arXiv [Deep Learning with Low Precision by Half-wave Gaussian Quantization](https://arxiv.org/abs/1702.00953)    
- 2017 ICLR [Loss-aware Binarization of Deep Networks](https://openreview.net/pdf?id=S1oWlN9ll)  
- 2017 ICLR [Trained Ternary Quantization](https://openreview.net/pdf?id=S1_pAu9xl)  
- 2017 ICLR [Incremental Network Quantization: Towards Lossless CNNs with Low-precision Weights](https://openreview.net/pdf?id=HyQJ-mclg)  
- 2016 arXiv [Accelerating Deep Convolutional Networks using low-precision and sparsity](https://arxiv.org/abs/1610.00324)  
- 2016 ECCV [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/pdf/1603.05279.pdf)  
- 2016 ICMLW [Overcoming Challenges in Fixed Point Training of Deep Convolutional Networks](https://arxiv.org/pdf/1607.02241.pdf)  
- 2016 ICML [Fixed Point Quantization of Deep Convolutional Networks](http://jmlr.org/proceedings/papers/v48/linb16.pdf)  
Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)  
- 2016 arXiv [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](http://arxiv.org/abs/1602.02830)  
- 2016 ICLR [Neural Networks with Few Multiplications](https://arxiv.org/abs/1510.03009)  
- 2015 NIPS [Backpropagation for Energy-Efficient Neuromorphic Computing](https://papers.nips.cc/paper/5862-backpropagation-for-energy-efficient-neuromorphic-computing.pdf)  
- 2015 NIPS [BinaryConnect: Training Deep Neural Networks with binary weights during propagations](https://papers.nips.cc/paper/5647-binaryconnect-training-deep-neural-networks-with-binary-weights-during-propagations.pdf)  
- 2015 ICMLW [Bitwise Neural Networks](http://minjekim.com/papers/icml2015_mkim.pdf)  
- 2015 ICML [Deep Learning with Limited Numerical Precision](http://www.jmlr.org/proceedings/papers/v37/gupta15.pdf)  
- 2015 ICLRW [Training deep neural networks with low precision multiplications](https://arxiv.org/abs/1412.7024)    
- 2015 arXiv [Training Binary Multilayer Neural Networks for Image Classification using Expectation Backpropagation](https://arxiv.org/abs/1503.03562)   
- 2014 NIPS [Expectation Backpropagation: Parameter-Free Training of Multilayer Neural Networks with Continuous or Discrete Weights](https://papers.nips.cc/paper/5269-expectation-backpropagation-parameter-free-training-of-multilayer-neural-networks-with-continuous-or-discrete-weights.pdf)  
- 2013 arXiv [Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation](https://arxiv.org/pdf/1308.3432.pdf)  
- 2011 NIPSW [Improving the speed of neural networks on CPUs](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37631.pdf)  


## Parameter Pruning  
- 2017 ICLR [Soft Weight-Sharing for Neural Network Compression](https://openreview.net/pdf?id=HJGwcKclx)  
- 2017 ICLR [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://openreview.net/pdf?id=SJGCiw5gl)  
- 2017 ICLR [Pruning Filters for Efficient ConvNets](https://openreview.net/pdf?id=rJqFGTslg)  
- 2016 arXiv [Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Pruning](https://arxiv.org/abs/1611.05128)  
- 2016 arXiv [Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures](https://arxiv.org/abs/1607.03250)  
- 2016 NIPS [Learning the Number of Neurons in Deep Networks](https://rsu.forge.nicta.com.au/people/jalvarez/LNN/AlvarezSalzmannNIPS16.pdf)  
- 2016 NIPS [Learning Structured Sparsity in Deep Learning](https://arxiv.org/abs/1608.03665) \[[code](https://github.com/wenwei202/caffe/tree/scnn)\]  
- 2016 NIPS [Dynamic Network Surgery for Efficient DNNs](http://128.84.21.199/abs/1608.04493)  
- 2016 ECCV [Less is More: Towards Compact CNNs](https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-319-46493-0_40/MediaObjects/419976_1_En_40_MOESM1_ESM.pdf)  
- 2016 CVPR [Fast ConvNets Using Group-wise Brain Damage](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Lebedev_Fast_ConvNets_Using_CVPR_2016_paper.pdf)  
- 2016 ICLR [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](http://arxiv.org/abs/1510.00149)  
- 2016 ICLR [Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications](http://arxiv.org/abs/1511.06530)
- 2015 arXiv [Structured Pruning of Deep Convolutional Neural Networks](http://arxiv.org/abs/1512.08571)  
- 2015 IEEE Access [Channel-Level Acceleration of Deep Face Representations](http://ieeexplore.ieee.org/document/7303876/)  
- 2015 BMVC [Data-free parameter pruning for Deep Neural Networks](http://arxiv.org/abs/1507.06149)
- 2015 ICML [Compressing Neural Networks with the Hashing Trick](http://jmlr.org/proceedings/papers/v37/chenc15.pdf)   
- 2015 ICCV [Deep Fried Convnets](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yang_Deep_Fried_Convnets_ICCV_2015_paper.pdf)  
- 2015 ICCV [An Exploration of Parameter Redundancy in Deep Networks with Circulant Projections](http://felixyu.org/pdf/ICCV15_circulant.pdf)  
- 2015 NIPS [Learning both Weights and Connections for Efficient Neural Networks](http://arxiv.org/abs/1506.02626)    
- 2015 ICLR [FitNets: Hints for Thin Deep Nets](http://arxiv.org/pdf/1412.6550v4.pdf)  
- 2014 arXiv [Compressing Deep Convolutional Networks
using Vector Quantization](http://arxiv.org/abs/1412.6115)  
- 2014 NIPSW [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)   
- 1995 ISANN [Evaluating Pruning Methods](http://publications.idiap.ch/downloads/papers/1995/thimm-pruning-hop.pdf)  
- 1993 T-NN [Pruning Algorithms--A Survey](http://axon.cs.byu.edu/~martinez/classes/678/Papers/Reed_PruningSurvey.pdf)  
- 1989 NIPS [Optimal Brain Damage](http://yann.lecun.com/exdb/publis/pdf/lecun-90b.pdf)  

## Transfer Learning  
- 2016 arXiv [What makes ImageNet good for transfer learning?](https://arxiv.org/abs/1608.08614)  
- 2014 NIPS [How transferable are features in deep neural networks?](https://arxiv.org/pdf/1411.1792v1.pdf)  
- 2014 CVPR [CNN Features off-the-shelf: an Astounding Baseline for Recognition](http://www.cv-foundation.org/openaccess/content_cvpr_workshops_2014/W15/papers/Razavian_CNN_Features_Off-the-Shelf_2014_CVPR_paper.pdf)  

## Hardware
- 2017 FPGA [Can FPGAs Beat GPUs in Accelerating Next-Generation Deep Neural Networks](http://jaewoong.org/pubs/fpga17-next-generation-dnns.pdf)  
- 2015 NIPS Tutorial [High-Performance Hardware for Machine Learning](https://media.nips.cc/Conferences/2015/tutorialslides/Dally-NIPS-Tutorial-2015.pdf)  
