to train the model to use a style image: 
python nst.py train --dataset dataset/ --batch-size 14 --style-image style.jpg --save-model-dir saved/style --epochs 5 --cuda 1

note that the dataset directory structure should be: <folder>/<folder>/images. like coco dataset, dataset/train2014/images


to use the trained style:
python nst.py eval --content-image louvre.jpg --model saved_styles/starrynight.model  --output-image generated.jpg --cuda 1  


This comes from pytorch examples in git: https://github.com/pytorch/examples



