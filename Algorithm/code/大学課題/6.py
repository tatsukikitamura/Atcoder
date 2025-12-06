import matplotlib.pyplot as plt

def object_trajectory(x):
    location = 10*x-5*x**2

    return location

xs = [x/100 for x in range(201)]
ys = [object_trajectory(x) for x in xs]

plt.plot(xs,ys)
plt.title('The Trajectory of a Thrown Object')
plt.xlabel('Horizontal Postion of Object')
plt.ylabel('Vertical Postion of Object')
plt.axhline(y = 0)
plt.show()

