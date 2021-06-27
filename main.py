import numpy as np
import matplotlib.patches as patches
import cv2
import scipy.stats as st
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
import scipy.optimize
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate
import math
W=2
H=4
w=W/2
h=H/2
Horizontal=1
half_Horizontal=0.5
Longitudinal=2
dict={
    'g':(0,0),
    'v':(0,-Longitudinal),
    'h':(Horizontal,0),
    'j':(2*Horizontal,0),
    'k':(3*Horizontal,0),
    'l':(4*Horizontal,0),
    'f':(-Horizontal,0),
    'd':(-2*Horizontal,0),
    's':(-3*Horizontal,0),
    'a':(-4*Horizontal,0),
    'c':(-Horizontal,-Longitudinal),
    'x':(-2*Horizontal,-Longitudinal),
    'z':(-3*Horizontal,-Longitudinal),
    'b':(Horizontal,-Longitudinal),
    'n':(2*Horizontal,-Longitudinal),
    'm':(3*Horizontal,-Longitudinal),
    'y':(0.5,Longitudinal),
    'u':(0.5+Horizontal,Longitudinal),
    'i':(0.5+2*Horizontal,Longitudinal),
    'o':(0.5+3*Horizontal,Longitudinal),
    'p':(0.5+4*Horizontal,Longitudinal),
    't':(-0.5,Longitudinal),
    'r':(-0.5-Horizontal,Longitudinal),
    'e':(-0.5-2*Horizontal,Longitudinal),
    'w':(-0.5-3*Horizontal,Longitudinal),
    'q':(-0.5-4*Horizontal,Longitudinal)
}



def normal_distribution(x):
    mean=np.array(dict[x])
    conv=np.array([[0.01,-0.01],[-0.01,0.01]])
    axis = np.random.multivariate_normal(mean=mean, cov=conv, size=1)
    list=axis[0].tolist()
    return list

def mid_point(x,y):
    x1=x[0]
    x2=x[1]
    y1=y[0]
    y2=y[1]
    b=(y2-x2)/(y1-x1)
    c= np.random.normal((-0.5/360)*6.28,(11.59/360)*3.14)
    newx=x1-y1
    newx2=x2-y2
    tanB=math.tan(c)
    ang=(b+tanB)/(1-b*tanB)
    temp=1/b
    midx=((newx2)/2+temp*(newx)/2)/(ang+temp)
    midy=ang*midx
    return (midx+y1,midy+y2)


def min_jerk(pos=None, dur=None, sen=None,vel=None, acc=None, psg=None):

    N = pos.shape[0]					# number of point
    D = pos.shape[1]					# dimensionality

    if not vel:
        vel = np.zeros((2,D))			# default endpoint vel is 0
    if not acc:
        acc = np.zeros((2,D))			# default endpoint acc is 0

    t0 = np.array([[0],[dur]])

    if not psg:					# passage times unknown, optimize
        if N > 2:
            psg = np.arange(dur/(N-1), dur-dur/(N-1)+1, dur/(N-1)).T
            func = lambda psg_: mjCOST(psg_, pos, vel, acc, t0)
            psg = scipy.optimize.fmin(func = func, x0 = psg)
        else:
            psg = []
    X=[]
    Y=[]
    X_point=[]
    Y_point=[]
    trj = mjTRJ(psg, pos, vel, acc, t0, dur)
    Z=[]
    i=1
    for x in trj:
        X.append(x[0])
        Y.append(x[1])
        Z.append(i)
        i=i+1
    for x in pos:
        X_point.append(x[0])
        Y_point.append(x[1])
        Z.append(i)
        i=i+1
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlim(-6,6)
    ax.set_ylim(-5,5)
    ax.plot(X,Y)
    for x,y in zip(X_point,Y_point):
        ax.scatter(x,y)
    ax.set_title(sen,fontsize=12,color='r')
    currentAxis = plt.gca()
    generate_keyboard(currentAxis)
  #  plt.show()
    plt.savefig('D:/picture/22.jpg',figsize=[3,10])
    return trj, psg

################################################################
###### Compute jerk cost
################################################################

def mjCOST(t, x, v0, a0, t0):

    N = max(x.shape)
    D = min(x.shape)

    v, a = mjVelAcc(t, x, v0, a0, t0)
    aa   = np.concatenate(([a0[0][:]], a, [a0[1][:]]), axis = 0)
    aa0  = aa[0:N-1][:]
    aa1  = aa[1:N][:]
    vv   = np.concatenate(([v0[0][:]], v, [v0[1][:]]), axis = 0)
    vv0  = vv[0:N-1][:]
    vv1  = vv[1:N][:]
    tt   = np.concatenate((t0[0]   , t, t0[1]   ), axis = 0)
    T    = np.diff(tt)[np.newaxis].T*np.ones((1,D))
    xx0  = x[0:N-1][:]
    xx1  = x[1:N][:]

    j=3*(3*aa0**2*T**4-2*aa0*aa1*T**4+3*aa1**2*T**4+24*aa0*T**3*vv0- \
         16*aa1*T**3*vv0 + 64*T**2*vv0**2 + 16*aa0*T**3*vv1 - \
         24*aa1*T**3*vv1 + 112*T**2*vv0*vv1 + 64*T**2*vv1**2 + \
         40*aa0*T**2*xx0 - 40*aa1*T**2*xx0 + 240*T*vv0*xx0 + \
         240*T*vv1*xx0 + 240*xx0**2 - 40*aa0*T**2*xx1 + 40*aa1*T**2*xx1- \
         240*T*vv0*xx1 - 240*T*vv1*xx1 - 480*xx0*xx1 + 240*xx1**2)/T**5

    J = sum(sum(abs(j)));

    return J

################################################################
###### Compute trajectory
################################################################

def mjTRJ(tx, x, v0, a0, t0, P):

    N = max(x.shape)
    D = min(x.shape)
    X_list = []

    if len(tx) > 0:
        v, a = mjVelAcc(tx, x, v0, a0, t0)
        aa   = np.concatenate(([a0[0][:]],  a, [a0[1][:]]), axis = 0)
        vv   = np.concatenate(([v0[0][:]],  v, [v0[1][:]]), axis = 0)
        tt   = np.concatenate((t0[0]   , tx, t0[1]   ), axis = 0)
    else:
        aa = a0
        vv = v0
        tt = t0

    ii = 0
    for i in range(1,int(P)+1):
        t = (i-1)/(P-1)*(t0[1]-t0[0]) + t0[0]
        if t > tt[ii+1]:
            ii = ii+1
        T = (tt[ii+1]-tt[ii])*np.ones((1,D))
        t = (t-tt[ii])*np.ones((1,D))
        aa0 = aa[ii][:]
        aa1 = aa[ii+1][:]
        vv0 = vv[ii][:]
        vv1 = vv[ii+1][:]
        xx0 = x[ii][:]
        xx1 = x[ii+1][:]

        tmp = aa0*t**2/2 + t*vv0 + xx0 + t**4*(3*aa0*T**2/2 - aa1*T**2 + \
                                               8*T*vv0 + 7*T*vv1 + 15*xx0 - 15*xx1)/T**4 + \
              t**5*(-(aa0*T**2)/2 + aa1*T**2/2 - 3*T*vv0 - 3*T*vv1 - 6*xx0+ \
                    6*xx1)/T**5 + t**3*(-3*aa0*T**2/2 + aa1*T**2/2 - 6*T*vv0 - \
                                        4*T*vv1 - 10*xx0 + 10*xx1)/T**3
        X_list.append(tmp)

    X = np.concatenate(X_list)

    return X

################################################################
###### Compute intermediate velocities and accelerations
################################################################

def mjVelAcc(t, x, v0, a0, t0):

    N = max(x.shape)
    D = min(x.shape)
    mat = np.zeros((2*N-4,2*N-4))
    vec = np.zeros((2*N-4,D))
    tt = np.concatenate((t0[0], t, t0[1]), axis = 0)

    for i in range(1, 2*N-4+1, 2):

        ii = int(math.ceil(i/2.0))
        T0 = tt[ii]-tt[ii-1]
        T1 = tt[ii+1]-tt[ii]

        tmp = [-6/T0, -48/T0**2, 18*(1/T0+1/T1), \
               72*(1/T1**2-1/T0**2), -6/T1, 48/T1**2]

        if i == 1:
            le = 0
        else:
            le = -2

        if i == 2*N-5:
            ri = 1
        else:
            ri = 3

        mat[i-1][i+le-1:i+ri] = tmp[3+le-1:3+ri]
        vec[i-1][:] = 120*(x[ii-1][:]-x[ii][:])/T0**3 \
                      + 120*(x[ii+1][:]-x[ii][:])/T1**3

    for i in range(2, 2*N-4+1, 2):

        ii = int(math.ceil(i/2.0))
        T0 = tt[ii]-tt[ii-1]
        T1 = tt[ii+1]-tt[ii]

        tmp = [48/T0**2, 336/T0**3, 72*(1/T1**2-1/T0**2), \
               384*(1/T1**3+1/T0**3), -48/T1**2, 336/T1**3]

        if i == 2:
            le = -1
        else:
            le = -3

        if i == 2*N-4:
            ri = 0
        else:
            ri = 2

        mat[i-1][i+le-1:i+ri] = tmp[4+le-1:4+ri]
        vec[i-1][:] = 720*(x[ii][:]-x[ii-1][:])/T0**4 \
                      + 720*(x[ii+1][:]-x[ii][:])/T1**4

    T0 = tt[1] - tt[0]
    T1 = tt[N-1]-tt[N-2]
    vec[0][:] = vec[0][:] +  6/T0*a0[0][:]    +  48/T0**2*v0[0][:]
    vec[1][:] = vec[1][:] - 48/T0**2*a0[0][:] - 336/T0**3*v0[0][:]
    vec[2*N-6][:] = vec[2*N-6][:] +  6/T1*a0[1][:]    -  48/T1**2*v0[1][:]
    vec[2*N-5][:] = vec[2*N-5][:] + 48/T1**2*a0[1][:] - 336/T1**3*v0[1][:]

    avav = inv(mat).dot(vec)
    a = avav[0:2*N-4:2][:]
    v = avav[1:2*N-4:2][:]

    return v, a


def dis(x,y):
    return (x[0]-y[0])**2+(x[1]-y[1])**2

def generate_keyboard(currentAxis):
    rect1=patches.Rectangle((-0.5,-1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect2=patches.Rectangle((-1.5,-1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect3=patches.Rectangle((-2.5,-1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect4=patches.Rectangle((-3.5,-1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect5=patches.Rectangle((-4.5,-1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect6=patches.Rectangle((0.5,-1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect7=patches.Rectangle((1.5,-1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect8=patches.Rectangle((2.5,-1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect9=patches.Rectangle((3.5,-1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect10=patches.Rectangle((-1,1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect11=patches.Rectangle((-2,1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect12=patches.Rectangle((-3,1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect13=patches.Rectangle((-4,1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect14=patches.Rectangle((-5,1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect15=patches.Rectangle((0,1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect16=patches.Rectangle((1,1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect17=patches.Rectangle((2,1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect18=patches.Rectangle((3,1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect19=patches.Rectangle((4,1),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect20=patches.Rectangle((-0.5,-3),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect21=patches.Rectangle((-1.5,-3),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect22=patches.Rectangle((-2.5,-3),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect23=patches.Rectangle((-3.5,-3),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect24=patches.Rectangle((0.5,-3),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect25=patches.Rectangle((1.5,-3),1,2,linewidth=2,edgecolor='r',facecolor='none')
    rect26=patches.Rectangle((2.5,-3),1,2,linewidth=2,edgecolor='r',facecolor='none')
    currentAxis.text(0,0,"g")
    currentAxis.text(0,-2,"v")
    currentAxis.text(1,0,"h")
    currentAxis.text(2,0,"j")
    currentAxis.text(3,0,"k")
    currentAxis.text(4,0,"l")
    currentAxis.text(-1,0,"f")
    currentAxis.text(-2,0,"d")
    currentAxis.text(-3,0,"s")
    currentAxis.text(-4,0,"a")
    currentAxis.text(1,-2,"b")
    currentAxis.text(2,-2,"n")
    currentAxis.text(3,-2,"m")
    currentAxis.text(-1,-2,"c")
    currentAxis.text(-2,-2,"x")
    currentAxis.text(-3,-2,"z")
    currentAxis.text(0.5,2,"y")
    currentAxis.text(1.5,2,"u")
    currentAxis.text(2.5,2,"i")
    currentAxis.text(3.5,2,"o")
    currentAxis.text(4.5,2,"p")
    currentAxis.text(-0.5,2,"t")
    currentAxis.text(-1.5,2,"r")
    currentAxis.text(-2.5,2,"e")
    currentAxis.text(-3.5,2,"w")
    currentAxis.text(-4.5,2,"q")
    currentAxis.add_patch(rect1)
    currentAxis.add_patch(rect2)
    currentAxis.add_patch(rect3)
    currentAxis.add_patch(rect4)
    currentAxis.add_patch(rect5)
    currentAxis.add_patch(rect6)
    currentAxis.add_patch(rect7)
    currentAxis.add_patch(rect8)
    currentAxis.add_patch(rect9)
    currentAxis.add_patch(rect10)
    currentAxis.add_patch(rect11)
    currentAxis.add_patch(rect12)
    currentAxis.add_patch(rect13)
    currentAxis.add_patch(rect14)
    currentAxis.add_patch(rect15)
    currentAxis.add_patch(rect16)
    currentAxis.add_patch(rect17)
    currentAxis.add_patch(rect18)
    currentAxis.add_patch(rect19)
    currentAxis.add_patch(rect20)
    currentAxis.add_patch(rect21)
    currentAxis.add_patch(rect22)
    currentAxis.add_patch(rect23)
    currentAxis.add_patch(rect24)
    currentAxis.add_patch(rect25)
    currentAxis.add_patch(rect26)

sentence=input()
index=0
print(len(sentence))
temp=[]
while(index<len(sentence)):
    point=normal_distribution(sentence[index])
    print(point)
    temp.append(point)
    index=index+1
i=0
new_list=[]
new_list.append(temp[0])
while(i<len(temp)-1):
    x=mid_point(temp[i],temp[i+1])
    if dis(temp[i],temp[i+1])>6:
        new_list.append(x)
    new_list.append(temp[i+1])
    i=i+1
print(new_list)
i=0

new_array=np.array(new_list)
print(new_array)
a,b=min_jerk(new_array,10000,sentence)

img_path=r"./1.jpg"
'''
img=cv2.imread(img_path)
plt.savefig('1.jpg')
print(img.shape)
cv2.rectangle(img,(100,0),(200,100),(0,0,255),10,shift=2)
cv2.rectangle(img,(1,-3),(-3,1),(255,255,0),10,shift=2)
cv2.imwrite('1.jpg',img)
cv2.imshow("fff",img)
'''

#rect=patches.Rectangle((-1,-2),2,4,linewidth=2,edgecolor='r',facecolor='none')

#currentAxis.add_patch(rect)
