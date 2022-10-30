import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
import scipy
import pylab
from scipy.stats import gaussian_kde
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from operator import itemgetter
import numpy as np
from scipy import interpolate
from scipy.interpolate import Rbf

matplotlib.rc('font', **{'family': 'DejaVu Sans', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{siunitx}')

font = {'size' : 12}
plt.rc('font', **font)
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
matplotlib.rcParams['axes.titlesize'] = 'medium'
matplotlib.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.labelcolor']='black'
plt.rcParams['xtick.color']='black'
plt.rcParams['ytick.color']='black'
plt.rcParams['legend.edgecolor']='black'
plt.rcParams['lines.linewidth'] = .75


default_colours = ['black',u'#E24A33', u'#348ABD', u'#988ED5', u'#777777', u'#FBC15E', u'#8EBA42', u'#FFB5B8', u'#CB04A5']


matplotlib.rc('axes', prop_cycle=(cycler('color', default_colours)))


cmap = plt.get_cmap('Spectral_r')


def suplabel(axis,label,label_prop=None,
             labelpad=5,
             ha='center',va='center'):
    ''' Add super ylabel or xlabel to the figure
    Similar to matplotlib.suptitle
    axis       - string: "x" or "y"
    label      - string
    label_prop - keyword dictionary for Text
    labelpad   - padding from the axis (default: 5)
    ha         - horizontal alignment (default: "center")
    va         - vertical alignment (default: "center")
    '''
    fig = pylab.gcf()
    xmin = []
    ymin = []
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        ymin.append(ax.get_position().ymin)
    xmin,ymin = min(xmin),min(ymin)
    dpi = fig.dpi
    if axis.lower() == "y":
        rotation=90.
        x = xmin-float(labelpad)/dpi
        y = 0.5
    elif axis.lower() == 'x':
        rotation = 0.
        x = 0.5
        y = ymin - float(labelpad)/dpi
    else:
        raise Exception("Unexpected axis: x or y")
    if label_prop is None: 
        label_prop = dict()
    pylab.text(x,y,label,rotation=rotation,
               transform=fig.transFigure,
               ha=ha,va=va,
               **label_prop)
    
def figsize(scale,two_column):
    if two_column == True :
        fig_width_pt = 246.0
    else :
        fig_width_pt = 510.0
    
    inches_per_pt = 0.0138889                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*0.8              # height in inches
#     fig_height = fig_width
    fig_size = [fig_width,fig_height]
    return fig_size

