# Part 1: Tensors in PyTorch
	
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