import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager

Xc = np.array([-1, 2, 3, 4])
Yc = np.array([-2, 3, 4, 5])
plt.plot(Xc,Yc)
plt.title("Matplotlib Test")
plt.show()

print(matplotlib.matplotlib_fname())
print(matplotlib.get_cachedir())
