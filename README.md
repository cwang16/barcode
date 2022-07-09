# Using Deep Learning to Improve Detection and Decoding of Barcodes
##
__The proposed pipline__ detects barcode, rotate barcode and debluring barcode images using RDD (Zhong et al., 2020) and deblurGAN-V2 (Kupyn et al., 2019). The PyTorch implementation of RDD, available at https://github.com/Capino512/pytorch-rotation-decoupled-detector, was used to detect and rotate barcode. The models are trained with around 2000 images, including images from Muenster, ArteLab (w/ AF), ArteLab (w/o AF), OpenFoodFacts, ZXing-ean13,and Skku inyong DB. The PyTorch implementation of DeblurGANv2, available at https://github.com/VITA-Group/DeblurGANv2, was used to deblur the barcode image. The models are trained with around 2000 images, including our own bounding box annotation. Those datasets made available in this repository to enable easy training or fine-tuning of models for other sets of barcode images. 
<br />
<br />
![Barcode Image](https://github.com/cwang16/barcode/blob/main/pipeline.png) <br /> 
<br />
<br />
## Content of the Repository
__barcode_image_debluring_dataset.zip__ contains the image name and source dataset for training RDD model  <br />
__oriented_barcode_detection_dataset.zip__ contains the bounding box annotation, image name and source dataset for traning deblurGANv2 model <br />

