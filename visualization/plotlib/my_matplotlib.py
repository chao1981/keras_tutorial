import matplotlib.pyplot as plt
import numpy as np

plt.subplot(111)
plt.axis([0, 5, 0, 100])
t = np.arange(0., 5., 0.2)
plt.plot(t, t**2, label="curve1")
plt.plot(t, t**3, label="curve2")
plt.legend()
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.grid(True)##显示网格
plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),arrowprops=dict(facecolor='black', shrink=0.05),)##显示箭头
plt.text(3, 90, r'$\mu=100,\ \sigma=15$') ##显示文字
plt.savefig('picture.png', format='png',dpi=600)##保存图片
plt.show()

#####################################################################################
##将图片生成gif
def generate_gif(imagesfolder,output='output.gif'):
  import imageio
  import os
  images = []
  filenames=os.listdir(imagesfolder)
  for filename in filenames:
    if (filename.endswith('.png')):
      images.append(imageio.imread(imagesfolder+filename))
  imageio.mimsave(output, images, duration=0.5)##duration=为每祯播放的时间单位s

#####################################################################################

##将图片生成视频
def images2video(imagesfolder="./images/",outputfile="output.avi"):
  import cv2
  from cv2 import VideoWriter_fourcc
  import os
  image=cv2.imread(imagesfolder+'0.png')##one picture in images to get the size of video
  print(len(image[0][0]))
  fourcc = VideoWriter_fourcc(*"MJPG")
  videoWriter = cv2.VideoWriter(outputfile, fourcc, 5, (len(image[0]), len(image)))
  filenames=os.listdir(imagesfolder)
  for filename in filenames:
    if (filename.endswith('.png')):
      frame = cv2.imread(imagesfolder +filename)
      videoWriter.write(frame)
  videoWriter.release()

#####################################################################################

##boxplot
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")  ##"dark","white","ticks"
data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
sns.boxplot(data=data)
sns.despine(left=True)  ##去除刻标
plt.show()
#####################################################################################

##kdeplot
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

x, y = np.random.multivariate_normal([0, 0], [[1, -.5], [-.5, 1]], size=300).T
pal = sns.dark_palette("palegreen", as_cmap=True)
sns.kdeplot(x, y, cmap=pal)
plt.show()
#####################################################################################

##heatmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

uniform_data = np.random.rand(10, 12)
flights = sns.load_dataset("flights")
flights = flights.pivot("month", "year", "passengers")
print(type(flights))
print(flights)
ax = sns.heatmap(flights)
# ax = sns.heatmap(flights, annot=True, fmt="d")
# ax = sns.heatmap(flights, linewidths=.5)
# ax = sns.heatmap(flights, cmap="YlGnBu")
# ax = sns.heatmap(flights, center=flights.loc["January", 1955])
plt.show()
exit()
#####################################################################################


##画饼状图
import matplotlib.pyplot as plt
labels='frogs','hogs','dogs','logs'
sizes=15,20,45,10
colors='yellowgreen','gold','lightskyblue','lightcoral'
explode=0,0.1,0,0
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
plt.axis('equal')
plt.show()
#####################################################################################