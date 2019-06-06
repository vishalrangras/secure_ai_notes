# Part 1: Tensors in PyTorch
	
* Sigmoid of x: 
```python
1/1+torch.exp(-x)
```
* To set random seed: ```torch.manual_seed(n)	# n is a natural number```
* To generate a tensor of given dimension having random normal values: ```torch.randn((1,5)) #Generates a tensor with one row and five columns```