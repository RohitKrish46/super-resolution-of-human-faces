# Super-Resolution-of-Human-Faces
This project showcases the integration of the Fast Super-Resolution CNN (FSRCNN) model onto the Atlas 200 DK platform for real-time facial image enhancement. By leveraging Face Detection models, we achieve efficient image capturing and processing, with potential applications including deployment on smartphones for scalable image enhancement with minimal data loss. The FSRCNN model significantly boosts image quality, offering faster super-resolution compared to traditional methods, making it ideal for real-time applications.

# Faster Super Resolution CNN
The Faster Super Resolution CNN, an advancement over SRCNN, addresses the limitations of its predecessor for real-time applications. While SRCNN effectively scales images, its reliance on operations like Bicubic interpolation and non-linear mapping hampers real-time performance, achieving only 1.6 frames per second. In contrast, FSRCNN achieves remarkable speed improvements, reaching 24 frames per second with minimal architectural changes. By eliminating Bicubic interpolation and optimizing the non-linear mapping phase, FSRCNN ensures efficient super-resolution without compromising performance, making it the preferred choice for real-time applications.

# Model Conversion
The Da Vinci architecture is the required format for utilizing models on the Ascend 310 chip. Configuration of the generated model can be accomplished through either the MindStudio GUI or the OMG command-line interface. While the GUI provides a user-friendly interface, the CLI offers greater flexibility in model conversion, making it preferable for advanced users seeking more customization options.

# Face Detection using Atlas 200 DK
The Atlas 200 AI Developer Kit harnesses the formidable processing power of the Ascend 310 processor, empowering AI developers to seamlessly deploy pre-trained models and conduct real-time testing of applications. With its integrated external camera, the Atlas 200 DK captures video data in real time, enabling precise face detection and seamless display of results through a presenter server interface.
# Results
This image is the output obtained from the Face Detection Model built on Atlas 200 DK: 
 ![](Atlas200DK/FSRCNN-DK/out/final.png) 
 
 This image is the final output obtained from the Da Vinci Model after integrating FSRCNN onto Atlas 200 DK:
 ![](Atlas200DK/FSRCNN-DK/out/20200425215019/0/SaveFilePostProcess_1/davinci_final_output_0_NHWC_output_0.jpeg) 
# References
* Chao Dong, Chen Change Loy, Xiaoou Tang. Accelerating the Super-Resolution Convolutional Neural Network, in Proceedings of European Conference on Computer Vision (ECCV), 2016
* https://github.com/Saafke/FSRCNN_Tensorflow
* https://towardsdatascience.com/review-fsrcnn-super-resolution-80ca2ee14da4
* https://www.huaweicloud.com/intl/en-us/ascend/doc/Atlas200DK/1.3.0.0/en/en-us_topic_0173402133.html
* https://www.huaweicloud.com/intl/en-us/ascend/doc/Atlas200DK/1.3.0.0/en/en-us_topic_0165968579.html

