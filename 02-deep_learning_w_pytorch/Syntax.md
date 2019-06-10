# Part 1: Tensors in PyTorch

*[Back to Home Page](https://github.com/vishalrangras/secure_ai_notes)*
	
* Sigmoid of x: 
```python
1/1+torch.exp(-x)
```
* To set random seed: 
```python
torch.manual_seed(n)			# n is a natural number
```
* To generate a tensor of given dimension having random normal values: 
```python
torch.randn((1,5))				#Generates a tensor with one row and five columns
```
* To create a random tensor of same shape as a source tensor
```python
torch.randn_like(source_tensor)
```
* To get sum of all the elements of a tensors
```python
torch.sum(source_tensor)		#You can also apply .sum directly on a tensor.
```
* To reshape or resize a tensor
```python
source_tensor.reshape(a, b)		#Returns a new tensor with shape a, b
source_tensor.resize_(a, b) 	#Returns the same tensor with shape a, b. In-place operation.
source_tensor.view(a, b)		#Returns a new tensor with shape a, b
```
* To do Matrix Multiplication
```python
torch.mm(source_tensor_1, source_tensor_2)
```
* To do in-place scalar multiplication
```python
source_tensor.mul_(k)			#Here, k is a scalar constant
```
* To create a random numpy array
```python
np_array = np.random.rand(a, b)		#Will create an array with dimensions of a, b with random values
```
* To create a tensor from numpy array
```python
dest_tensor = torch.from_numpy(a)	#The memory is shared between Numpy Array and Tensor, hence they are the same objects, only different representations.
```
* To create a numpy array from a tensor
```python
dest_np_array = source_tensor.numpy()
```

*[Back to Home Page](https://github.com/vishalrangras/secure_ai_notes)*

# Part 2: Neural Networks with PyTorch

### TORCHVISION.TRANSFORMS
> Transforms are common image transformations. They can be chained together using `Compose`. Additionally, there is the `torchvision.transforms.functional` module which gives fine-grained control over the transformations. This is useful if you have to build a more complex transformation pipeline (e.g. Segmentation tasks)

```python
# Here, Compose composes several transforms together
# toTensor converts a PIL image or numpy.ndarray to a tensor
# Normalize normalizes a tensor with given mean and given standard deviation.
transform = transforms.Compose([transforms.toTensor(),
								transforms.Normalize((0.5,),(0.5,)),])

```

### TORCHVISION.DATASETS
> All datasets are subclasses of `torch.utils.data.Dataset` and they have __getitem__ and __len__ methods implemented. They all can be passed to a `torch.utils.data.DataLoader` which can load multiple samples parallelly using `torch.multiprocessing` workers.

Following datasets are available with Torchvision: MNIST, Fashion-MNIST, KMNIST, EMNIST, FakeData, COCO, LSUN, ImageFolder, DatasetFolder, ImageNet, Cifar, STL10, SVHN, PhotoTour, SBY, Flickr, VOC, Cityscapes, SBD.

All datasets have two common arguments i.e. transform (for input transformation) and target_transform (for output transformation)

```python							
# Download MNIST dataset to a given location and transform it using provided transform. 
# Flag train is set to True in order to indicate that data is for training purposes and not for test purposes.
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)

# Load the training data with the batch size of 64 and shuffle the order
# torch.utils.data.DataLoader combines a dataset and a sampler (in our case, we our using shuffling instead of a sampler).
# It also provides a single- or multi-process iterator over the dataset.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
```

#### Analyzing the dataset
Our MNIST dataset which is now loaded through DataLoader in our `trainloader` variable can be iterated through a for loop as follows:

```python
for image, label in trainloader:
	## Operation on a single image and it's label
``` 

or alternatively, a single batch of datapoint can be extracted using Python's iterator as follows:

```python
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)
```

#### To print the image itself:
```python
# Here, numpy.squeeze() is used to remove single dimensional entries from the shape of an array.
# In our case, that single dimension entry indicates a single channel image i.e. Grayscale image.
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')

# To understand operation of squeeze:
print(images[1].numpy().shape)					# Will return (1, 28, 28)
print(images[1].numpy().squeeze().shape)		# Will return (28, 28)
```

### Now, the task of building a Neural Network:

Our goal here is to build a simple fully-connected neural network (also known as dense network) having an input layer, a hidden layer and an output layer.
Each unit in one layer is connected to each unit in next layer. We will build this network using simple matrix multiplication.

In FCN (Fully Connected Networks), the input to each layer must be a one-dimensional vector. Our images are 2D tensors of 28 x 28, so we need to convert them into a 1D vector having 784 (28 * 28) elements instead. Also, each batch of our data contains 64 images, hence our original dimension of `images` tensor is [64, 1, 28, 28]. We need to flatten or reshape this tensor to have shape as [64, 784], i.e. 64 vectors or 64 images (datapoints) each having 784 elements in it.

Basically, we are stacking our 64 images of a batch into a 2D tensor.

We need to implement following steps in order to build a neural network:

1. Flattening the batch of images.

```python
print(images.shape)		# Will return [64, 1, 28, 28]
inputs = images.view(images.shape[0], -1)
print(inputs.shape)		# Will return [64, 784]
```

*[What is the meaning of parameter -1](https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch)*?
-1 is a way of telling the library: "Give me a tensor that has given numbers of columns/rows and you compute the appropriate number of rows/columns that is necessary to make this happen".