# DRST Baseline

## Original Paper
Distributionally Robust Learning for Unsupervised Domain Adaptation

## Problem
The target data is unlabeled and may look different from the labeled source data.

## Main Idea
The paper uses two neural networks:

1. Classification network
   - predicts the class of an image.

2. Domain discrimination network
   - predicts whether an image comes from the source or target domain.
   - this is used to estimate a density ratio.

## DRST
The model:

1. Predicts target images.
2. Calculates confidence.
3. Selects the most confident target samples.
4. Gives them pseudo-labels.
5. Adds them to training data.
6. Trains again.
7. Repeats.

## Our Goal
First reproduce the original DRL/DRST method.

After reproduction, analyze weaknesses and develop Adaptive-DRST.
