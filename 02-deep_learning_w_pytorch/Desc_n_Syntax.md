# Part 1: Tensors in PyTorch

*[Back to Home Page](https://github.com/vishalrangras/secure_ai_notes)*

### So what are Neurons anyway?

> Deep Learning is based on artificial neural networks which have been around in some form since the late 1950s. The networks are built from individual parts approximating neurons, typically called units or simply "neurons." Each unit has some number of weighted inputs. These weighted inputs are summed together (a linear combination) then passed through an activation function to get the unit's output.

Mathematically this looks like: 

$$
\begin{align}
y &= f(w_1 x_1 + w_2 x_2 + b) \\
y &= f\left(\sum_i w_i x_i +b \right)
\end{align}
$$

With vectors this is the dot/inner product of two vectors:

$$
h = \begin{bmatrix}
x_1 \, x_2 \cdots  x_n
\end{bmatrix}
\cdot 
\begin{bmatrix}
           w_1 \\
           w_2 \\
           \vdots \\
           w_n
\end{bmatrix}
$$

//TODO: Add image of Neuron calculation here including how Activation takes place

//TODO: Also add the intuition of parallelism for Deep Learning from deeplizard reference.

### What are Tensors?

> It turns out neural network computations are just a bunch of linear algebra operations on *tensors*, a generalization of matrices. A vector is a 1-dimensional tensor, a matrix is a 2-dimensional tensor, an array with three indices is a 3-dimensional tensor (RGB color images for example). The fundamental data structure for neural networks are tensors and PyTorch (as well as pretty much every other deep learning framework) is built around tensors.
	
* Sigmoid of x: 
```python
1/1+torch.exp(-x)
```

* Creating tensors and some operations on them:
 
```python
# Set the random seed so things are predictable
torch.manual_seed(n)			# n is a natural number

# To create tensors with Standard Normal Distribution
torch.randn((1,5))				#Generates a tensor with one row and five columns
torch.randn_like(source_tensor) #To create a random tensor of same shape as a source tensor

# To get sum of all the elements of a tensors
torch.sum(source_tensor)		#You can also apply .sum directly on a tensor.

# To reshape or resize a tensor
source_tensor.reshape(a, b)		#Returns a new tensor with shape a, b
source_tensor.resize_(a, b) 	#Returns the same tensor with shape a, b. In-place operation.
source_tensor.view(a, b)		#Returns a new tensor with shape a, b

# To do Matrix Multiplication
torch.mm(source_tensor_1, source_tensor_2)

# To do in-place scalar multiplication
source_tensor.mul_(k)			#Here, k is a scalar constant

# To create a random numpy array
np_array = np.random.rand(a, b)		#Will create an array with dimensions of a, b with random values

# To create a tensor from numpy array
dest_tensor = torch.from_numpy(a)	#The memory is shared between Numpy Array and Tensor, hence they are the same objects, only different representations.

# To create a numpy array from a tensor
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

### Neural Network Diagram

// TODO: Add Network Diagram here

### Flow Diagram

// TODO: Add a flow diagram explain above implemented steps

In FCN (Fully Connected Networks), the input to each layer must be a one-dimensional vector. Our images are 2D tensors of 28 x 28, so we need to convert them into a 1D vector having 784 (28 * 28) elements instead. Also, each batch of our data contains 64 images, hence our original dimension of `images` tensor is [64, 1, 28, 28]. We need to flatten or reshape this tensor to have shape as [64, 784], i.e. 64 vectors or 64 images (datapoints) each having 784 elements in it.

Basically, we are stacking our 64 images of a batch into a 2D tensor.

We need to implement following steps in order to build a neural network:

#### 1. Flattening the batch of images:

```python
print(images.shape)								# Will return [64, 1, 28, 28]
inputs = images.view(images.shape[0], -1)
print(inputs.shape)								# Will return [64, 784]
```

*[What is the meaning of parameter -1](https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch)*?

-1 is a way of telling the library: "Give me a tensor that has given numbers of columns/rows and you compute the appropriate number of rows/columns that is necessary to make this happen".

#### 2. Define number of Neurons in each Layer and hence defining the Neural Network:

* Since, our input features are 784 (28 * 28), number of neurons in our input layer are going to be 784.

* As per the requirement given in our exercise notebook, we need a hidden layer with 256 neurons.

* And, we need output layer to have 10 neurons, each neuron should be giving the softmax probability of hand-written digit image having value from 0 to 9.

```python
n_input = inputs.shape[1]
n_hidden = 256
n_output = 10
```

* Next, we need to create Weight and Bias matrices/tensors for our hidden layer as well as output layer. We also need to randomly populate values into this matrices/tensors such that they have Standard Normal Distribution with mean=0 and std=1.

> The order of a given Weight Matrix is determined by `number of inputs coming x number of outputs going`. Thus, for hidden layer weight matrix, its dimension is going to be 784 x 256, while for output layer matrix, its going to be 256 x 10.

> The order of a given Bias Matrix is determined by `number of columns for respective matrix x 1`, since we are going to add Bias matrix into Weight matrix.

```python
torch.manual_seed(7)

W1 = torch.randn(n_input, n_hidden)
B1 = torch.randn(n_hidden)

W2 = torch.randn((n_hidden, n_output))
B2 = torch.rand(n_output)
```

#### 3. Activation for Hidden Layer and Output Layer:

* We are instructed to use Sigmoid as Activation function for Hidden Layer, while to use Softmax Function for Output Layer.

```python
def sigmoid_activation(x):
    return 1/(1+torch.exp(-x))
	
y1 = sigmoid_activation(torch.mm(inputs,W1)+B1)
print("y1: "+str(y1.shape))

out = torch.mm(y1, W2)+B2 # output of your network, should have shape (64,10)
print("Output: "+str(out.shape))


def softmax(x):
    ## TODO: Implement the softmax function here
	# dim=1 just says that the sum operation is done across columns.
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1,1)

# Here, out should be the output of the network in the previous excercise with shape (64,10)
probabilities = softmax(out)

# Does it have the right shape? Should be (64, 10)
print(probabilities.shape)
# Does it sum to 1?
print(probabilities.sum(dim=1))	
```