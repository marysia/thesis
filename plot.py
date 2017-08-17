import matplotlib as mpl
mpl.use('TkAgg')
# import matplotlib.pyplot as plt
import pylab as plt

x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sens = [0.88, 0.94, 0.97, 0.98, 0.99, 0.99, 1.0, 1.0, 1.0, 1.0, 0.0]
acc = [0.87, 0.93, 0.97, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
fprate = [0.16, 0.08, 0.03, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

plt.plot(x, acc, label='accuracy')
plt.plot(x, sens, label='sensitivity')
plt.plot(x, fprate, label='fp-rate')
plt.xticks(x)
plt.yticks(x)
plt.legend(loc='best')
plt.title('My pretty plot')
plt.show()