import torch
import  numpy as np

tensor_from_list = torch.tensor([1,2,3,4]) #从列表创建
print(tensor_from_list) #tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])


numpy_array = np.array([4,5,6,7,8,9,10,11])  # tensor([ 4,  5,  6,  7,  8,  9, 10, 11], dtype=torch.int32) 只是在终端显示的信息会更多一些 如果是交互界面后面信息没有

print(numpy_array) #[ 4  5  6  7  8  9 10 11]

tensor_from_list1 = torch.from_numpy(numpy_array)
torch_from_tensor = torch.tensor(numpy_array)
print(tensor_from_list1) #tensor([ 4,  5,  6,  7,  8,  9, 10, 11], dtype=torch.int32)

print(torch_from_tensor) #tensor([ 4,  5,  6,  7,  8,  9, 10, 11], dtype=torch.int32)


zeros_tensor = torch.zeros((4,4)) #创建一个对应矩阵  如果参数只有一个 那就是创建一个对应参数量的全零列表 tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
print(zeros_tensor)
one_tensor = torch.ones((3,3)) #全一

range_tensor = torch.arange(0,10,2) #包含左边不包含右边 第三个参数是间隔
print(range_tensor) #tensor([0, 2, 4, 6, 8])

uniform_tensor = torch.rand((3,4)) #随机生成一个 3行4列矩阵 大小为0-1
print(uniform_tensor)

normal_tensor = torch.randn((3,2)) #随机生成一个符合标准正态分布的张量
print(normal_tensor)

uninitialized_tensor = torch.empty((2,2)) #不会初始化张量值 取决于当时内容的内容
print(uninitialized_tensor)

like_tensor = torch.ones_like(zeros_tensor) #创建一个和现有张量 形状和数据类型一样的张量 所有元素初始化为1 那么类似的也有torch.zeros_like 等
print(like_tensor)

element = normal_tensor[1] #获取到了normal_tensor变量第二个内容 normal_tensor 是一个3*2的 所以获取到的是一个 1*2的内容
print(element)

sliced_tensor = normal_tensor[0:2] #左开右闭
print(sliced_tensor)

#这些是关于张量形状的操作
shape = sliced_tensor.shape #查看张量的形状
print(shape)

#修改张量的形状
reshaped_tensor = sliced_tensor.view((1,4)) #这个函数要求新形状元素总数和之前的一样
print(reshaped_tensor)

transposed_tensor = reshaped_tensor.t() #转置操作
print(transposed_tensor)

#数学运算

sum_tensor = reshaped_tensor + transposed_tensor #这个操作要求两个张量的形状相同 但是在这里不同却可以输出 是因为广播机制 如果两个张量形状不同 其中一个张量的维度为1 另一个不是 那么维度为1的张量复制自己的内容保证和另一个张量的维度相同
print(sum_tensor)

broadcasted_tensor = reshaped_tensor * 3 #这个只是将每个元素乘了一遍 乘3
print(broadcasted_tensor)

producted_tensor = torch.matmul(reshaped_tensor, transposed_tensor)
print(producted_tensor)