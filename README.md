# Defense against Adversarial Attack

**Dataset** : [NIPS 2017 competition flower102 dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)\
**Base Model** : [Pytorch-Flower-Image-Classification](https://github.com/MohammadBakir/Pytorch-Flower-Image-Classification)\
**Denoise Method** : [
DPIR
](https://github.com/cszn/DPIR/tree/master)

## Execute description
Download all needed file and put it into the direction that you execute the code.\
Then follow the steps to execute :
1. Get base model by executing **flower_training.ipynb**, and make sure the accuracy in testing set is good enough. 
2. Generate the preprocess dataset by executing **preprocessing.ipynb**. In this step, you will need denoise model and use it to restore both adversarial examples and original examples. Make sure the model that use for generating adversarial examples is the best one in previous step.
3. Train adversarial model by executing **flower_adv_training.ipynb**
4. Evaluate adversarial model by executing **test.ipynb**, the testing data can be generated by executing **preprocessing_testset.ipynb**