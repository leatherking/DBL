# Implementation of "Dual Balanced Class-Incremental Learning with im-Softmax and Angular Rectification"

## Abstract

Owing to the superior performances, exemplar-based methods with knowledge distillation are widely applied in class incremental learning. However, it suffers from two drawbacks: (i) Data imbalance between the old/learned and new classes causes the bias of the new classifier towards the head/new classes. (ii) Deep neural networks suffer from distribution drift when learning sequence tasks, which results in narrowed feature space and deficient representation of old tasks.For the first problem, we analyze the insufficiency of softmax loss when dealing with the problem of data imbalance in theory and then propose the imbalance softmax (im-softmax) loss to relieve the imbalanced data learning, where we re-scale the output logits to underfit the head/new classes.For another problem, we calibrate the feature space by incremental-adaptive angular margin~(IAAM) loss. The new classes form a complete distribution in feature space yet the old are squeezed. To recover the old feature space, we first compute the included angle of normalized features and normalized anchor prototypes, and use the angle distribution to represent the class distribution, then we replenish the old distribution with the deviation from the new. Each anchor prototype is predefined as a learnable vector for a designated class. The proposed im-softmax reduces the bias in the linear classification layer. IAAM rectifies the representation learning, reduces the intra-class distance, and enlarges the inter-class margin.Finally, we seamlessly combine the im-softmax and IAAM in an end-to-end training framework, called the Dual Balanced Class Incremental Learning (DBL), for further improvements.Experiments demonstrate the proposed method achieves state-of-the-art performances on several benchmarks, such as CIFAR10, CIFAR100, Tiny-ImageNet, and ImageNet-100.

## How to use

```
pip install -r requirements.txt
python main.py --config=./exps/dbl_cifar.json
```

## Acknowledge

Our code is based on [PyCIL](https://github.com/G-U-N/PyCIL).
