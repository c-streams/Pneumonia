# Diagnosing Pediatric Pneumonia Using Convolutional Neural Networks

## Executive Summary

Medical imaging, such as X-rays, are important tools used to help diagnose certain conditions. The limited availability of highly trained doctors who can interpret these images, however, can lead to a slow diagnostic process, delaying needed treatment. Childhood pneumonia is one such condition that when diagnosed early can be life saving. According to the World Health Organization, pneumonia is responsible for 16% of child deaths under 5 years old [1]. In an attempt to address this problem, I have built an image based deep learning model to diagnose pediatric pneumonia from chest X-rays. It is my hope that such a model could be utilized to expedite the disease screening process and serve as a second opinion to trained physicians.

I obtained 5,000+ physician labeled pediatric chest X-rays from a 2018 study by Kermany et al. who developed a generalized artificial intelligence system to diagnose conditions such as macular degeneration, diabetic retinopathy, and pneumonia [2]. The X-rays where obtained from children ages 1-5.

Using Keras with a TensorFlow backend, I built newly trained Convolutional Neural Networks (CNNs) and compared them to a transfer learning approach with the VGG16 and InceptionV3 models trained on the ImageNet dataset to address both binary and multi-class classification problems. Binary classification was defined as identifying normal chest X-rays from any type of pneumonia and the multi-class classification problem as distinguishing normal X-rays from bacterial and viral pneumonia cases. The models were trained on an AWS Deep Learning AMI (Ubuntu 16.4) Version 11.0 GPU enabled computer with 4 virtual CPUs and 61 GiB memory.

Optimizing for sensitivity with a target specificity over 65%, my binary classification CNN performed the best with a AUC-ROC score of 0.96, 88% accuracy, 99% sensitivity, and 71% specificity. These results are close to that of human experts. I suspect that my model performed better than the transfer learning approaches due to its simplicity. Since neural networks inherently overfit, a simpler model architecture will reduce complexity which can cause overfitting. Optimizing for the same metrics, the multi-class VGG16 model performed the best with a AUC-ROC score of 0.95, 82% accuracy, 99% sensitivity, and 66% specificity. Identification of viral pneumonia was consistently lower in all 3 multi-class models. I suspect this is due to the small sample size available. I believe additional data is required to improve the performance of the 3 class models.

To conclude, my results demonstrate the feasibility of using a CNN to diagnose pneumonia from chest X-rays with a respectable accuracy. Future exploration is required to further improve these models as well as replicate the findings of Kermany et al.


### Introduction

Pneumonia is a type of acute respiratory infection impacting the lungs that is caused by a variety of infectious agents such as bacteria, viruses, and fungi [1]. When afflicted with pneumonia, alveoli in the lungs fill with fluid and pus making it difficult to breath and prevent the intake of oxygen (Figure 1). In children, pneumonia is the largest infectious cause of death worldwide, responsible for 16% of deaths under 5 years old [1]. The most common treatment for bacterial pneumonia is antibiotics while for viral pneumonia is supportive care and antiviral medication [2]. If left untreated, pneumonia can be fatal. Because different types of pneumonia require different treatments, chest X-rays are an important tool used to help distinguish types in addition to presentable symptoms. Early detection and treatment is critical to reducing pneumonia fatalities in children.

<img src="https://github.com/c-streams/Pneumonia/blob/master/images/pne.png" width="40%" height="50%">

Limited availability of highly trained doctors who can interpret chest X-rays, however, leads to slow diagnostic processes, delaying much needed treatment. This is especially true in south Asia and sub-Saharan Africa where rates of pneumonia are much higher [1].

### Artificial Intelligence as Clinical Decision Support Systems

Artificial intelligence (AI) has the potential to revolutionize the field of radiology and transform image based medical diagnostics and treatment. While even highly trained physicians may see hundreds of thousands of X-rays in the course of their training and career, a machine learning model can be trained on millions. Interpretability and reliability, however, remain challenges for such AI diagnostic systems [2]. These challenges notwithstanding, these systems can still play an important role as a support system for human experts in expediting disease screening processes and serving as a second opinion (Figure 2).

<img src="https://github.com/c-streams/Pneumonia/blob/master/images/ai.png" width="40%" height="50%">

Armed with AI systems, studies have shown that physicians make more accurate predictions, leading to better patient outcomes [3].

### Problem Statement

My goal was to build a convolution neural network from scratch (i.e. newly trained weights) to diagnose pneumonia from pediatric chest X-rays and evaluate its performance against a transfer learning approach with the prebuilt VGG16 and InceptionV3 Keras models trained on the ImageNet dataset.

<img src="https://github.com/c-streams/Pneumonia/blob/master/images/problem.png" width="40%" height="50%">

I split this project into two parts, defined by a binary classification problem and a multi-class classification problem. Identifying pneumonia from normal cases and then distinguishing bacterial and viral pneumonia from normal cases (Figure 3).

### Dataset

I utilized a dataset from a recent study by Kermany et al. who developed a generalized artificial intelligence system to diagnose conditions such as macular degeneration, diabetic retinopathy, and pneumonia [2]. In this study, over 5,000 X-rays where obtained from children ages 1-5 at Guangzhou Women and Children’s Medical Center, Guangzhou, China during regular clinical visits. The images were labeled by two expert physicians and verified by a third physician.

The dataset  is composed of a train and test folder, each with normal and pneumonia sub-folders. Within the condition type folders there are jpeg chest X-ray images. In the test data, there are 390 pneumonia X-rays and 234 normal. In the train data, there are 3,883 pneumonia X-rays and 1,349 normal. The pneumonia images are further classified by their type, bacterial or viral, in their file names. Considering the relatively small size of the dataset, I utilized data augmentation techniques during modeling.

### Image Analysis and Processing 

In order to determine the amount of preprocessing required before modeling, I investigated the images themselves. Sample X-rays of each class are shown below in Figure 4.

<img src="https://github.com/c-streams/Pneumonia/blob/master/images/xrays.png" width="40%" height="50%">

From the X-rays above, it is not immediately obvious to the untrained eye which cases are which. Luckily machine learning is here to help!

I examined the class balances in the dataset and discovered that there are more cases of pneumonia in my dataset than normal ones. There are more cases of bacterial pneumonia in particular. Unbalanced classes are not uncommon with medical data. Classification algorithms perform better with an even ratio of classes otherwise the minority class is typically underrepresented as a rare event. I therefore balanced the classes by oversampling (sampling with replacement) the minority class.

Next I examined how many color channels were in the images, as well as their sizes and aspect ratios in an effort to inform my image standardization choice. Neural networks require the same image sizes. All images in the train and test have 3 color channels, but have varying aspect ratios and sizes. I decided to standardize the images to 224 x 224 since this is a requirement for the VGG16 model I decided to use.

Curious about the color distribution differences in the images, I plotted color histograms for each of the classes (Figure 5).  While there does appear to be differences in the distribution in each class, there are not consistent. I plotted different samples and got widely different results. The only consistent trend was a large peak at pixel 0 or black, which is to be expected from X-rays.

<img src="https://github.com/c-streams/Pneumonia/blob/master/images/colorhist.png" width="40%" height="50%">

After standardizing the images, I next investigated data augmentation to increase the overall size of my training set in order to improve the performance of my CNN model. I used the Keras Image Augmentation API, which generates images in real time during the model fitting process. Considering the application of my end model (to diagnose X-rays images), I chose augmentation parameters that are appropriate for variations we might see in X-rays. These include shifting the width and height of the X-rays. Supplementing the training dataset with a variety of image positions will help improve the generalization of my model so that it is not trained on a specific kind of positioning.

### Modeling Overview 

I created CNNs from scratch (i.e. with newly trained weights) and evaluated their performance against a transfer learning approach with the prebuilt VGG16 and InceptionV3 Keras CNN models trained on the ImageNet dataset.

Training a CNN from scratch (starting with a random initialization of weights) is often rare in practice because it is rare to get a large enough dataset to train. Transfer learning is the process of using pre-trained weights or extracted features from a pre-trained network (Figure 6). These pre-trained networks typically are very deep networks that have been trained on datasets composed of millions of images. These deep networks can take weeks to train using vast amounts of computing power. Rather than reinventing the wheel, we can use these networks as a basis to classify similar images since the high level features will be similar.

<img src="https://github.com/c-streams/Pneumonia/blob/master/images/transfer.png" width="40%" height="50%">

In order to speed up the time to train these deep networks, the models were trained on an AWS Deep Learning AMI (Ubuntu 16.4) Version 11.0 GPU enabled computer with 4 virtual CPUs and 61 GiB memory.

### CNN Model & Evaluation

After some trial and error, I decided to build a simple binary CNN with 3 convolution/pooling layers and 2 dense layers in order to minimize the number of parameters. I increased the filter size with each convolution layer in order to gradually identify more details from the x-rays. In order to improve the performance of my model, I implemented a callback to reduce the learning rate of the model when the test loss does not improve by a specified amount. Reducing the learning rate is known to improve model performance when learning plateaus. The final model performed the best for the binary problem. It achieved a AUC-ROC score of 0.96, 88% accuracy,  99% sensitivity, and 71% specificity. From Figure 7, you can see that the number of false negatives (5) has been minimized.

<img src="https://github.com/c-streams/Pneumonia/blob/master/images/matrix1.png" width="40%" height="50%">

The receiver operator curve (ROC) explains the balance between the true positive and true negative rates (Figure 8). The AUC-ROC score is the area under this curve, which at its best is 1.0.  With a score of 0.96, my model did well.

<img src="https://github.com/c-streams/Pneumonia/blob/master/images/roc1.png" width="40%" height="50%">

While it achieved good metrics, the model was still relatively overfit as demonstrated by tracking the train and validation loss and accuracy scores during training (Figure 9). The large separation between the test and validation curve indicates overfitting.

<img src="https://github.com/c-streams/Pneumonia/blob/master/images/cnn.png" width="40%" height="50%">

### Transfer Learning VGG16 Model & Evaluation

I selected the prebuilt Keras VGG16 deep learning model trained on the ImageNet dataset for my first transfer learning model [4]. ImageNet is open source image dataset with tens of millions of images of various objects. Models trained on this dataset tend to generalize very well to many different types of images content. I chose the VGG16 model because compared to others available in Keras, VGG16 has competitive accuracy scores on the validation ImageNet set with the lowest depth. I suspect that using more layers might overfit my use case. I only used the bottom of the VGG16 model architecture with the associated pre-trained weights and built a dense network on top to classify my dataset with the appropriate labels. This greatly limits the number of parameters that need to be trained since the basic structures of the images have already been identified by the out of the box pre-trained weights.

The multi-class VGG16 model performed the best for this problem. While still overfit, it has respectable results. It achieved a AUC-ROC score of 0.95 with 82% accuracy, 99% sensitivity, and 66% specificity (Figure 10).  The number of false negatives (combined viral and bacterial false negatives) is 3 out of a 624 test set.

<img src="https://github.com/c-streams/Pneumonia/blob/master/images/matrix2.png" width="40%" height="50%">

For a multi-class problem, each class has its own ROC curve as shown in Figure 11. The average AUC-ROC score was 0.95.

<img src="https://github.com/c-streams/Pneumonia/blob/master/images/roc2.png" width="40%" height="50%">

From Figure 12, it can be seen that the VGG16 model also experienced some over fitting. The train and validation loss scores actually move in the opposite direction.

<img src="https://github.com/c-streams/Pneumonia/blob/master/images/model.png" width="40%" height="50%">


### Transfer Learning InceptionV3 Model & Evaluation

I also performed transfer learning with the Keras InceptionV3 model in order to try and replicate the results of the Kermany et al. study [2]. It is important to note that their study did not attempt the 3 class problem, but performed another binary comparison to distinguish bacterial from viral pneumonia.

I found the InceptionV3 to have the worst performance for both binary and multi-class problems.  I was able to achieve 69% and 50% accuracy for the binary and multi-class problems respectively.

### Discussion

All 6 models suffered from overfitting in that the train and test loss and accuracy scores greatly differed and in some cases moved in opposite directions in subsequent epochs of training. In the CNN models I built, I incorporated dropout to reduce some overfitting and only used one dense layer to minimize the number of parameters to be trained considering the relatively small size of the dataset. I also restricted one dense output layer for each transfer learning model for the same reason, to reduce the number of parameters to be trained.

I evaluated my models on various metrics, such as AUC-ROC, accuracy, precision, sensitivity, and specificity, however, I paid particular attention to sensitivity. In the case of diagnosing pneumonia, we want to minimize predicting false negatives, that is telling a patient they don’t have pneumonia when they do. The consequences of a false negative can result in no treatment, which can be deadly. Alternatively, false positives, telling a patient they have pneumonia when they are healthy, isn’t as damaging to the patient. In this case, a healthy patient would receive treatment, like antibiotics, that won’t impact their health as negatively. In the worst possible case, antibiotics are known to decrease the diversity of bacteria in the gut, which can lead to stress, depression, and obesity [5]. It is important to note that there should be a balance between sensitivity and specificity in the case of pneumonia because the over prescription of antibiotics has led to the rise of antibiotic resistant bacteria. In fact, studies have found that upwards of 40% of antibiotics prescribed for acute respiratory tract infections, like pneumonia, are unnecessary [5]. Should this current trend of over prescription continue, it is possible that the bacteria that cause various infectious diseases will be resistant to antibiotics. With this in mind, I designed my CNN to minimize sensitivity to a certain extent, without letting specificity drop below 65%.

Considering these metrics, my binary CNN performed the best with a AUC-ROC score of 0.96, 88% accuracy, 99% sensitivity, and 71% specificity. These results are close to that of human experts. I suspect that my model performed better than the transfer learning approaches due to its simplicity. Since neural networks inherently overfit, a simpler model architecture will reduce overfitting.

The multi-class VGG16 model performed the best for its problem.  While still overfit,  it had respectable results with a AUC-ROC score of 0.95, 82% accuracy, 99% sensitivity, and 66% specificity. These results are almost comparable to those of human experts. When looking at the bacterial and viral cases, predictions for viral were almost always worse. I believe this has to do with the smaller dataset for viral. It would seem that oversampling is not sufficient.

Interestingly enough, the InceptionV3 model performed the worst in both problems. Kermany et al. were able to achieve a 92% accuracy with this model for the binary classification problem, while I achieved 68% [2]. It is possible that they used different model hyperparameters, which were not made available in their publication. In order to replicate their exact study, I would need more details.

If choosing between both transfer learning models, I recommend VGG16 over InceptionV3 since it has a simpler architecture. The base InceptionV3 model is more complex than the VGG16 with 159 layers compared to the 23 of VGG16.

Nonetheless, I was able to reach my goal and demonstrate the feasibility of using CNNs and transfer learning to diagnose pneumonia from chest X-rays with respectable results.

### Next Steps

In terms of next steps, I would like to further explore different transfer learning base models. Keras has a handful of models, such as Xception and VGG19, that could also yield positive results. In addition, I would like to reach out to the point of contact for the Kermany et al. study to get a copy of their code and understand how they achieved their results using the InceptionV3 model. I replicated my InceptionV3 model as best I could with the information provided in the paper, but would like to take a look at the actual code for more details.

Since my multi-class classification models were not quite as good at differentiating pneumonia cases, I would like to attempt another binary classification to properly distinguish bacterial from viral pneumonia. Additional data would most likely be required since there were only 1000+ viral images in the current dataset, which I believe was the cause of consistent underperformance of my 3 models for identifying viral pneumonia.


### References

[1] World Health Organization,”Fact Sheet: Pneumonia”, (2016), http://www.who.int/en/news-room/fact-sheets/detail/pneumonia

[2] Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X- Ray Images for Classification”, Mendeley Data, v2, http://dx.doi.org/10.17632/rscbjbr9sj.2

[3] Kontzer, Tony (2017), “Deep Learning Drops Error Rate for Breast Cancer Diagnoses by 85%”, Nvidia, https://blogs.nvidia.com/blog/2016/09/19/deep-learning-breast-cancer-diagnosis/

[4] Simonyan, Karen & Zisserman, Andrew. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv 1409.1556.

[5] Fiore, David C. et al. “Antibiotic overprescribing: still a major concern.” Journal of Family Practice 66.12 (2017). https://www.mdedge.com/sites/default/files/Document/November-2017/JFP06612730.PDF

 

 

