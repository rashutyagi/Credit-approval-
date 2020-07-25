# Credit-approval-
In this project I have applied Support Vector Machine  Algorithm to classify based on the given features whether a particular application for credit card will get approved or not with 93% accuracy rate.
Dataset Link -> https://archive.ics.uci.edu/ml/datasets/credit+approval



# Important Points Session 2

npimg = np.sum(npimg/3, axis=-1) is faster than npimg = np.sum(npimg, axis=-1)/3 in terms of implementation

Because in first one we need to perform the same operation multiple times.

# Before 2012 it was our task to write the kernels on our own for identifying different shapes and objects which was really a tedious task. How will we write one kernel which can identify every single kind od Circle Shape ? It would take a PHD to find that kernel but a DNN can do it in minutes.

# Images are a collection(s) of channel(s). For black and white images, we have only one channel

# We add layers so that we can have receptive field atleast equal to the size of the image. Also we always prefer to break our image into components.

# Face recognition happens at 50\*50.

# But many papers start at 224\*224 but they do cheating that they reach till 56\*56 and then start the actual convolution.

# 400\*400 Is a huge huge resolution you can do crowd counting at this resolution.

# Bigger kernels are better but we don&#39;t have good hardware for them . 3\*3 is supported by nvidia.

# If you need to make something complex you can easily increase the number of kernels.

# Once you have normalized all the data then there is no problem but we never mix the normalised and unnormalized data that will create a lot of problems.

# We are going to divide our image into various very simple features and when we superimpose them we get our image back.

# Gradient means slow change in the pixel value.

# In order to get to know this edge in the below image we will need to see at least 11 pixels

![](RackMultipart20200725-4-1fb0tpf_html_c8a62bb56ea1b2c3.png)

# It means we will need a receptive field of 11 (Keep this number in mind for 400\*400 image). You need to see this number in the image of your data.

# We know the concept of Local Receptive field and Global receptive field but our main interest will be towards Global Receptive field only.

# 3\*3 kernel is also good because it has axis of symmetry

# Also note carefully in the below image that you cannot draw a curve in 2\*2 kernel but on a 3\*3 kernel it can be done (it will be only a line untill more than one 2\*2 is used)

![](RackMultipart20200725-4-1fb0tpf_html_3fed1edcf31e24bc.png)

# Also if our network would want 2\*2 it can be done using a 3\*3 only like this

![](RackMultipart20200725-4-1fb0tpf_html_18cb766a3c1523fe.png)

# GPU is designed in such a way that it can do one operation millions of times parallely.

![](RackMultipart20200725-4-1fb0tpf_html_1c20eded3f00fce4.png)

# What actually happens in convolution is something like this :

# Suppose you want to identify a feature like this

![](RackMultipart20200725-4-1fb0tpf_html_5fc4e2020919ed3a.png)

# Now this can be done by amplifying the exact feature which we want to identify and it can be also done by de-amplifying the nearby features also so we will need something like this over this

![](RackMultipart20200725-4-1fb0tpf_html_b87fd9721c90ee7f.png)

# Kernel tells us the intensity of a feature in our image.

# Negative numbers in kernel also helps to push the important features by keeping them away(by being negative)

# We filter out information by throwing un important information by making them negative.

![](RackMultipart20200725-4-1fb0tpf_html_29a21882b6c96ca5.png)

![](RackMultipart20200725-4-1fb0tpf_html_5055855448b6f1cd.png)

# We will nearly all the time use 2\*2 kernel for maxpooling

# When we take 200 layers for 400\*400 image we get RF of 400 at the output layer

![](RackMultipart20200725-4-1fb0tpf_html_cc5fc5fc3d9c60d7.png)

Yellow one will give better information than the blue one about the central pixel of the input image.

# This is a lie -\&gt; (But live with this lie till session 5) and the lie is -\&gt; When we use max pooling our receptive field Doubles.

# While using max pooling we lose features but what backpropagation does is it converts that loss like we have filtered features. So, we don&#39;t lose features we filter features.

# DNN are invariant to size means if we train on small image of dog it will identify the big dog also.

# During maxpooling we communicate to our network that keep those features large which one you want to identify the most which are the most important to you.

# 11\*11 is very specific to our 400\*400 image and the assumption that object size = image size. For every dataset we need to see by zooming the image like how many pixels do you need in order to identify an edge. This is something we should learn experimentally.

# You should not perform convolution on a channel of size 7\*7 and at max 5\*5. Keep this always in mind.

# Max pooling just filters out the relevant information and that relevant information is added by the Convolutional kernels.

# Some max pooling rules –

1.
# It should be away from the input layer
2.
# It should be away from the output layer as well

# Max Pooling acts like a filter and in order to let that filter act properly we will need to provide some convolutions to it beforehand.

# Suppose you are working on medical images where each and every detail matter then there we will need to use as many as 1024 kernels and always keep in mind that as you increase the number of kernels you will need to increase the dataset as well.

# These are the two architectures A and B which we have

![](RackMultipart20200725-4-1fb0tpf_html_3f222053cbdfa5c1.png)

# Between every layer we have Max-pooling and also keep in mind that the whole world is using Architecture B but we will use Architecture A for initial Sessions.

# We will use 3\*3 as it is accelerated

![](RackMultipart20200725-4-1fb0tpf_html_7eb1f5f651e14655.png)

![](RackMultipart20200725-4-1fb0tpf_html_4fed3a4ebcba688a.png)

# The last 512 is important for us actually not the ones previous to it and we should not have a massive jump for any of those like do not do something like 32-\&gt;32-\&gt;-64-\&gt;64-\&gt;512 we will not do something like this as well as in this there is lot of pressure on some layer to learn a lot.

# Number of channels in the convolving Kernel must be equal to the number of channels in the input.

# A kernel will never have 1 channel a kernel will always have as many channels as there are channels in the input image/immediate previous image.

# We will always use 3\*3 to increase the number of channels and never to reduce the number of channels.
