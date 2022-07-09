# Using Deep Learning to Improve Detection and Decoding of Barcodes
##
![Barcode Image](https://github.com/cwang16/barcode/blob/main/pipeline.png) <br /> 
<br />
<br />
__The proposed pipline__ detects barcode, rotate barcode and debluring barcode images using RDD (Zhong et al., 2020) and deblurGAN-V2 (Kupyn et al., 2019). The PyTorch implementation of RDD, available at https://github.com/Capino512/pytorch-rotation-decoupled-detector, was used to detect and rotate barcode. The models are trained with around 2000 images, including images from Muenster, ArteLab (w/ AF), ArteLab (w/o AF), OpenFoodFacts, ZXing-ean13,and Skku inyong DB. The PyTorch implementation of DeblurGANv2, available at https://github.com/VITA-Group/DeblurGANv2, was used to deblur the barcode image. The models are trained with around 2000 images, including our own bounding box annotation. We provided all the bounding box annotations, images' name and source dataset used in the paper to enable easy training or fine-tuning of models for other sets of barcode images. 
<br />

## Content of the Repository
__oriented_barcode_detection_dataset.zip__ contains the bounding box annotation, image name and source dataset for traning RDD model <br />
__barcode_image_debluring_dataset.zip__ contains the image name and source dataset for training deblurGANv2 model  <br />

## Source Datasets:
Muenster: https://www.uni-muenster.de/PRIA/forschung/index.shtml and https://github.com/rohrlaf/SlaRle.js
ArteLab (w/ AF) and ArteLab (w/o AF): http://artelab.dista.uninsubria.it/downloads.html
OpenFoodFacts: https://github.com/openfoodfacts/openfoodfacts-ai/issues/15
ZXing-ean13: https://github.com/zxing/zxing/tree/master/core/src/test/resources/blackbox
Skku inyong DB: http://dspl.skku.ac.kr/home_course/data/barcode/skku_inyong_DB.zip

