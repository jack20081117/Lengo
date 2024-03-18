from tkinter import *
import numpy as np
from copy import deepcopy
from scipy.signal import convolve2d
from numba import jit
import time

boardLength=9
size=30

F=G=M=np.zeros([boardLength,boardLength])
#F=2*np.random.random([boardLength,boardLength])-np.ones([boardLength,boardLength])

color=1
remaining=1.00


# def clear():global f;f=np.zeros([boardLength,boardLength])

def f2color(f):
    '''
    :param f:介于[-1,1]的值，为棋盘上一点的状态
    :return:十六进制下的颜色字符串
    '''
    f=np.max([np.min([f,1]),-1])
    if f>=0:
        R=int(234*(1-f))
        G=int(196*(1-f))
        B=int(134*(1-f))
    else:
        R=int(234-f*21)
        G=int(196-f*59)
        B=int(134-f*121)
    return f'#{R:02x}{G:02x}{B:02x}'


def g2color(g):
    '''
    :param g:介于[0,1]的值，为棋盘上一点的气
    :return:十六进制下的颜色字符串
    '''
    R=G=B=int(g*255)
    return f'#{R:02x}{G:02x}{B:02x}'


@jit(nopython=True)
def step(x): return (x+np.abs(x))/(2*x+0.001)  #阶跃函数


@jit(nopython=True)
def relu(x): return np.maximum(x,0)


@jit(nopython=True)
def fri(x):  #友好函数
    #return 0.5*x-0.5*np.abs(x)+1
    return np.exp(-5*relu(-x))


def laplacian(x):
    '''
    离散情况下，一个矩阵的拉普拉斯约等于一个卷积操作
    :param x:需要卷积的矩阵
    :return:卷积后的结果
    '''
    l=np.array([[1,2,1],
                [2,-12,2],
                [1,2,1]])
    return convolve2d(x,l,mode='same',boundary='wrap')


@jit(nopython=True)
def transform(f,g):
    '''
    棋盘状态关于气的转变
    :param f:棋盘状态矩阵
    :param g:气状态矩阵
    :return:更改后的f和g
    '''
    # if np.abs(f)>=g:
    #     f=f*g/np.abs(f)
    #f=f*g
    f=f*(1-(1-g)**3)
    return f,g


#进行向量化
vStep=np.vectorize(step)
vFri=np.vectorize(fri)
vTransform=np.vectorize(transform)


# def getg(f,t,h=0.01):
#     if t<=0:return np.ones([boardLength,boardLength])-abs(f)
# 
#     oldg=getg(t-h)
#     newg=deepcopy(oldg)
#     newg+=vStep(f)*vStep(laplacian(vFri(f)*oldg))
#     newg+=vStep(-f)*vStep(laplacian(vFri(-f)*oldg))
#     return newg

def getgByrk4(f,t,h=0.01):
    '''
    通过runge-kutta方法对微分方程进行求解
    :param f:棋盘状态矩阵
    :param t:步长
    :param h:粒度
    :return:通过f求出来的g(t)
    '''
    func=lambda t,g:(vStep(f)*vStep(laplacian(vFri(f)*g))+vStep(-f)*vStep(laplacian(vFri(-f)*g)))*(np.ones([boardLength,boardLength])-g)
    T=0
    g=np.ones([boardLength,boardLength])-np.abs(f)
    for _ in range(int(t/h)):
        k1=h*func(T,g)
        k2=h*func(T+0.5*h,g+0.5*k1)
        k3=h*func(T+0.5*h,g+0.5*k2)
        k4=h*func(T+h,g+k3)

        g+=(k1+2*k2+2*k3+k4)/6
        T+=h
    #maxg=np.max(g)
    #print(maxg)
    return vTransform(f,g)


def getmByrk4(f,t,h=0.01):
    func=lambda t,m:(1-vStep(-m*laplacian(m)))*laplacian(m)
    T=0
    m=deepcopy(f)

    for _ in range(int(t/h)):
        k1=h*func(T,m)
        k2=h*func(T+0.5*h,m+0.5*k1)
        k3=h*func(T+0.5*h,m+0.5*k2)
        k4=h*func(T+h,m+k3)

        m+=(k1+2*k2+2*k3+k4)/6
        T+=h

    return m


#与GUI交互


tk=Tk()
tk.resizable(width=False,height=False)
tk.geometry('%dx%d+0+0'%(3*size*boardLength+90,size*boardLength+100))
tk.title("Lengo")

# clearButton=Button(tk,text='clear',command=clear)
# clearButton.pack()
# clearButton.place(x=10,y=size*boardLength+10)
canvasF=Canvas(tk,width=size*boardLength,height=size*boardLength,bg='#eac486',bd=0,highlightthickness=0)
canvasF.place(x=0,y=0)

canvasG=Canvas(tk,width=size*boardLength,height=size*boardLength,bg='white',bd=0,highlightthickness=0)
canvasG.place(x=size*boardLength+30,y=0)

canvasM=Canvas(tk,width=size*boardLength,height=size*boardLength,bg='#eac486',bd=0,highlightthickness=0)
canvasM.place(x=2*size*boardLength+60,y=0)

ScoreLabel=Label(tk,text='test',fg='black')
ScoreLabel.place(x=size*boardLength+20,y=size*boardLength+10)

RemainingLabel=Label(tk,text='Turn: Black, Remaining: 0.00',fg='black')
RemainingLabel.place(x=0.1*size*boardLength,y=size*boardLength+70)

PresumptiveLabel=Label(tk,text='0.00',fg='black')
PresumptiveLabel.place(x=0.1*size*boardLength,y=size*boardLength+10)

Placer=Scale(tk,from_=0,to=100,orient=HORIZONTAL)
Placer.place(x=0.3*size*boardLength,y=size*boardLength+10)

def create(event):
    global F,G,M,color,remaining
    x,y=event.widget.winfo_pointerxy()
    deployed=remaining*color*Placer.get()/100
    # print(x,y)
    # print(x//size,y//size)
    if x>=boardLength*size+10 or y>=boardLength*size+30: return 0
    curval=F[(x-10)//size,(y-30)//size]
    #if abs(curval)>0.05:return 0
    if curval*color<-0.105: return 0
    if abs(curval+deployed)>1: return 0

    F[(x-10)//size,(y-30)//size]+=deployed
    remaining=(1-Placer.get()/100)*remaining
    colordict={1:'Black',-1:'White'}
    if remaining<=0:
        color=-color
        remaining=1
        F,G=getgByrk4(F,10)
        M=getmByrk4(F,10,h=0.01)

    RemainingLabel.config(text='Turn: %s, Remaining: %.2f'%(colordict[color],remaining))

def delete(event):
    global F
    x,y=event.widget.winfo_pointerxy()
    if x>boardLength*size or y>boardLength*size: return 0
    F[(x-10)//size,(y-30)//size]=0


canvasF.bind('<Button-1>',create)
#canvasF.bind('<B1-Motion>',create)
canvasF.bind('<Button-3>',delete)


#canvasF.bind('<B3-Motion>',delete)

def show(f,g,m,canvasF,canvasG,canvasM):
    '''
    对棋盘与气进行可视化
    :param f:棋盘状态矩阵
    :param g:气状态矩阵
    :param canvasF:画布F
    :param canvasG:画布G
    '''
    #画棋盘上的线
    ScoreLabel.config(
        text='Black Score: %.2f'%np.sum(relu(m))+' White Score: %.2f'%np.sum(relu(-m))+' Spread: %.2f'%np.sum(m)
    )
    for i in range(boardLength):
        canvasF.create_line(size*i+size/2,size/2,size*i+size/2,size*boardLength-size/2)
        canvasF.create_line(size/2,size*i+size/2,size*boardLength-size/2,size*i+size/2)
    #画棋盘上的点，只适用于19*19的棋盘
    # for i in range(3,16,6):
    #     for j in range(3,16,6):
    #         canvasF.create_oval(size*(i+1/3),size*(j+1/3),size*(i+2/3),size*(j+2/3),fill='black')
    for i in range(boardLength):
        for j in range(boardLength):
            canvasG.create_rectangle(size*i,size*j,size*i+size,size*j+size,fill=g2color(g[i,j]),outline='')
            canvasG.create_text(size*(i+1/2),size*(j+1/2),text='%.2f'%g[i,j])
            canvasM.create_rectangle(size*i,size*j,size*i+size,size*j+size,fill=f2color(m[i,j]),outline='')
            if np.abs(m[i,j])>0.01:
                canvasM.create_text(size*(i+0.5),size*(j+0.5),text='%.2f'%m[i,j],fill='white' if m[i,j]>0 else 'black')
            if np.abs(f[i,j])>0.05:
                canvasF.create_oval(size*i,size*j,size*i+size,size*j+size,fill=f2color(f[i,j]),outline='')
                canvasF.create_text(size*(i+1/2),size*(j+1/2),text='%.2f'%f[i,j],fill='white' if f[i,j]>0 else 'black')
    tk.update_idletasks()
    tk.update()


if __name__=='__main__':
    try:
        #print(getg(F,1))
        while True:
            show(F,G,M,canvasF,canvasG,canvasM)
            PresumptiveLabel.config(text='%.2f'%(remaining*(float(Placer.get())*0.01)))
            time.sleep(0.01)
            canvasF.delete('all')
            canvasG.delete('all')
            canvasM.delete('all')
    except TclError:
        pass
