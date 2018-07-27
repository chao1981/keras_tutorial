import torch
import torch.nn as nn
pool = nn.MaxPool2d(2, stride=2,return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)

input = torch.tensor([[[[1., 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]]])
output,index=pool(input)
output1=unpool(output,index)
output2=unpool(output,index,output_size=torch.Size([1, 1, 5, 5]))
print(input)
print(output)
print(output1)
print(output2)