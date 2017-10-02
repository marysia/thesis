
import matplotlib as mpl

mpl.use('Agg')
import pylab as plt

def create_plot():


    x_axis = [1000, 5000, 10000, 15000]
    y_axis = [.65, .70, .75, .80, .85]
    y_axis_z3 = [.723, .766, .781, .792]
    y_axis_b = [.769, .825, .828, .823]

    plt.plot(x_axis, y_axis_z3, label='Z3')
    plt.plot(x_axis, y_axis_b, label='B')
    plt.xticks(x_axis)
    plt.yticks(y_axis)

    plt.legend(loc='best')
    plt.title('tryout')
    plt.save('test.png')
