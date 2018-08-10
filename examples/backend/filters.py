import numpy as np

x=np.random.rand(4,128)
y=np.random.rand(128,32)

result=[]
result.append(np.sum(x[:2,:])/128/2)
result.append(np.sum(x[2:3,:])/128)
result.append(np.sum(x[3:,:])/128)

result=np.array(result)
result=(result-result.min())/(result.max()-result.min())
print(result)

result=[]
result.append(np.sum(np.dot(x[:2,:],y))/32/2/128)
result.append(np.sum(np.dot(x[2:3,:],y))/32/128)
result.append(np.sum(np.dot(x[3:,:],y))/32/128)
result=np.array(result)
result=(result-result.min())/(result.max()-result.min())
print(result)
