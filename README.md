# Neural-Style-Transfer-in-Pytorch
Pytorch implementation of artistic neural sty transfer in Jupyter notebook

This repo contains my port of Andrew Ng's Deep Learning Specialization's assignment on Neural styke transfer which was in tensorflow 2. I dislike tensorflow due to its
strange way of coding using the gradienttape. Though I completed the assignment on my own, I couldn't fully grasp the core concepts until I implemented it in pytorch 
and overcame those issues myself.

One of the strangest thing I encountered was that setting the vgg model to eval mode wasn't enough, I have to freeze the weights in the layers else it would be very slow 
during training. Another strange thing is the LARGE step size used for training. Due to the high loss, if the steps are small, like less than 1, it would take many steps 
to see the visual effects. 

I've also included the use of LBFGS optimizer for training, besides the usual Adam. It's like Adam with very large steps. Frankly I don't like it because it tends to 
fail easily during training and the output can't be observed during training. The result also tends to be quite blurry. I just included it here to show how it can be used. 

Using jupyter notebook enables runtime observation of the style transfer and makes it easier to use. Just specify the content and style images, layers to use, optimizer
type and it will run. You can of coz specify other parameters as well, such as weights for the style layers to be used, number of epochs, alpha and beta for content and 
style weights in computing total cost. 

I found that using the relu layers give better results compared to the conv layers. You can use as many style layers as you like, just modify the style_weights dictionary. 
Content layer is always just 1 layer. 

Hope this notebook helps somone to understand neural style transfer easily. 
