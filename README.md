# Pytorch Implementation of Facial Landmark Detection Models Trained on Synthetic Data

## **To Read Complete Report See CMPT_732_Project_Report.pdf**

## **Problem Statement**
The problem statement for this work was provided by Industrial Light & Magic (ILM). The goal was to design and train facial landmarks models on the provided Fake It Till You Make It (Fake-It) dataset from Microsoft (Wood et al., 2021), and to test the models on real world faces. These tasks were to be completed so that results could be efficiently computed on consumer-level GPUs. Two additional bonus tasks were also completed: running the best model in real time over webcam feed (30 fps) and identifying which data augmentation techniques offered the most improvements to model accuracy for this problem.

## **Experiments**
Two different subsets of training data were used to assess the minimum viable dataset size for crossing the domain gap: 1,000 and 10,000 images. 

Differing combinations of augmentation techniques were used to evaluate the effect these augmentations play in the ability for the model to learn. The models trained with no augmentations at all acted as a baseline. After initially testing all of the models with and without all augmentations included in this project, it was noted that for MobileNetv3 and ResNet50, the model version trained on augmented images resulted in worse (greater) NME values during testing. To explore this further a third subset of augmentations was introduced. This included just occlusion and translation, as it was hypothesized that the training data already showed sufficient examples with rotation, and blurring was not an overly relevant augmentation. Moreover, coordinate regression-based-methods using a single-step approach typically suffer from poor global accuracy - additional rotation might have confused these models more than helped them. These three subsets were evaluated and compared based on their results.

In total, there are 3 different data augmentation combinations: no augmentation, translation and occlusion only, and all augmentations (rotation, blur, translation and occlusion). This led to a total of 6 experiments for each model, the results of which are discussed in the next section. 



## **References**
Earp, S. W. F., Samacoits, A., Jain, S., Noinongyao, P., & Boonpunmongkol, S. (2021). Sub-pixel face landmarks using heatmaps and a bag of tricks. arXiv preprint. arXiv:2103.03059

He, K., Zhang, X., Ren, S., & Sun, J. (2015, December 10). Deep Residual Learning for Image Recognition. Proceedings of the IEEE conference on computer vision and pattern recognition, 770-778. https://doi.org/10.1109/cvpr.2016.90

Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019). Searching for MobilNetV3. Proceedings of the IEEE/CVF international conference on computer vision, 1314-1324. https://doi.org/10.48550/arXiv.1905.02244

Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., & Adam, H. (2017, April 17). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. Cornell University. https://doi.org/10.48550/arXiv.1704.04861

Jin, H., Liao, S., & Shao, L. (2021). Pixel-in-Pixel Net: Towards Efficient Facial Landmark Detection in the Wild. International Journal of Computer Vision, 129(12), 3174-3194. https://doi.org/10.1007/s11263-021-01521-4

Kim, T., Mok, J., & Lee, E. (2021, August 9). Detecting Facial Region and Landmarks at Once via Deep Network. Sensors (Basel). https://doi.org/10.3390/s21165360

Sagonas, C., Tzimiropoulos, G., Zafeiriou, S., & Pantic, M. (2013). 300 faces in-the-Wild Challenge: The first facial landmark localization challenge. 2013 IEEE International Conference on Computer Vision Workshops. 10.1109/iccvw.2013.59

Wood, E., Baltrusaitis, T., Hewitt, T., Dziadzio, S., Cashman, T. J., & Shotton, J. (2021). Fake it till you make it: Face analysis in the wild using Synthetic Data alone. 2021 IEEE/CVF International Conference on Computer Vision (ICCV). https://doi.org/10.1109/iccv48922.2021.00366

Zakharov, E., Shysheya, A., Burkov, E., & Lempitsky, V. (2019). Few-Shot Adversarial Learning of Realistic Neural Talking Head Models. Proceedings of the IEEE/CVF international conference on computer vision, 9459-9468. https://doi.org/10.1109/iccv.2019.00955
