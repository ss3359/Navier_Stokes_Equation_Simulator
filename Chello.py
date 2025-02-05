import math 
import numpy as np



#Define the grid
grid=np.zeros((20,20,20,3))
h = 0.5
g=9.81 #meters per second squared
P=20 # PSI


#Functions. 
def vx(x,y,z,t): 
    return 2*np.sin(np.radians(x))*np.cos(np.radians(y))*np.exp(-t)
def vy(x,y,z,t): 
    return 3*np.cos(np.radians(x))*np.sin(np.radians(z))*np.exp(-2*t)
def vz(x,y,z,t): 
    return 4*np.sin(np.radians(y))*np.sin(np.radians(z))*np.exp(-3*t)

def dvxdx(x,y,z,t): 
    return (vx(x+h,y,z,t)-vx(x,y,z,t))/h
def dvxdy(x,y,z,t): 
    return (vx(x,y+h,z,t)-vx(x,y,z,t))/h
def dvxdz(x,y,z,t): 
    return (vx(x,y,z+h,t)-vx(x,y,z,t))/h
def dvxdt(x,y,z,t): 
    return (vx(x,y,z,t+h)-vx(x,y,z,t))/h

def dvydx(x,y,z,t): 
    return (vy(x+h,y,z,t)-vy(x,y,z,t))/h
def dvydy(x,y,z,t): 
    return (vy(x,y+h,z,t)-vy(x,y,z,t))/h
def dvydz(x,y,z,t): 
    return (vy(x,y,z+h,t)-vy(x,y,z,t))/h
def dvydt(x,y,z,t): 
    return (vy(x,y,z,t+h)-vy(x,y,z,t))/h

def dvzdx(x,y,z,t): 
    return (vz(x+h,y,z,t)-vz(x,y,z,t))/h
def dvzdy(x,y,z,t): 
    return (vz(x,y+h,z,t)-vz(x,y,z,t))/h
def dvzdz(x,y,z,t): 
    return (vz(x,y,z+h,t)-vz(x,y,z,t))/h
def dvzdt(x,y,z,t): 
    return (vz(x,y,z,t+h)-vz(x,y,z,t))/h


# Second Derivatives 
def d2vxdxx(x,y,z,t):
    return (vx(x+h,y,z,t)-2*vx(x,y,z,t)-vx(x-h,y,z,t))/(h**2)
def d2vxdyy(x,y,z,t):
    return (vx(x,y+h,z,t)-2*vx(x,y,z,t)-vx(x,y-h,z,t))/(h**2)
def d2vxdzz(x,y,z,t):
    return (vx(x,y,z+h,t)-2*vx(x,y,z,t)-vx(x,y,z-h,t))/(h**2)


def d2vydxx(x,y,z,t):
    return (vy(x+h,y,z,t)-2*vy(x,y,z,t)-vy(x-h,y,z,t))/(h**2)
def d2vydyy(x,y,z,t):
    return (vy(x,y+h,z,t)-2*vy(x,y,z,t)-vy(x,y-h,z,t))/(h**2)
def d2vydzz(x,y,z,t):
    return (vy(x,y,z+h,t)-2*vy(x,y,z,t)-vy(x,y,z-h,t))/(h**2)

def d2vzdxx(x,y,z,t):
    return (vz(x+h,y,z,t)-2*vz(x,y,z,t)-vz(x-h,y,z,t))/(h**2)
def d2vzdyy(x,y,z,t):
    return (vz(x,y+h,z,t)-2*vz(x,y,z,t)-vz(x,y-h,z,t))/(h**2)
def d2vzdzz(x,y,z,t):
    return (vz(x,y,z+h,t)-2*vz(x,y,z,t)-vz(x,y,z-h,t))/(h**2)

def v(x,y,z,t):
    return (y,-x,t,-z)
def mu(x, y, z, t):
    #Constant viscosity (e.g., air ~ 1.8e-5 Pa.s)
    return 1.0

    # Viscosity varies with temperature
    # return 1.0 + 0.1 * np.sin(np.radians(x)) * np.cos(np.radians(y)) * np.exp(-t)

def rho(scalar): 
    return 1.225 #Constant Density of air kg/m^3

def LapX(x,y,z,t): 
    return d2vxdxx(x,y,z,t)+d2vxdyy(x,y,z,t)+d2vxdzz(x,y,z,t)
def LapY(x,y,z,t): 
    return d2vydxx(x,y,z,t)+d2vydyy(x,y,z,t)+d2vydzz(x,y,z,t)
def LapZ(x,y,z,t): 
    return d2vzdxx(x,y,z,t)+d2vzdyy(x,y,z,t)+d2vzdzz(x,y,z,t)

#For the NS Equation, we are going to solve for dPdx, dPdy, dPdz
def dPdx(x,y,z,t):
    scalar=dvxdt(x,y,z,t)+vx(x,y,z,t)*dvxdx(x,y,z,t)+vy(x,y,z,t)*dvxdy(x,y,z,t)+vz(x,y,z,t)*dvxdz(x,y,z,t)
    return -(rho(scalar)-mu(x,y,z,t)-g)
def dPdy(x,y,z,t):
    scalar=dvydt(x,y,z,t)+vx(x,y,z,t)*dvydx(x,y,z,t)+vy(x,y,z,t)*dvydy(x,y,z,t)+vz(x,y,z,t)*dvydz(x,y,z,t)
    return -(rho(scalar)-mu(x,y,z,t)-g)
def dPdz(x,y,z,t):
    scalar=dvzdt(x,y,z,t)+vx(x,y,z,t)*dvzdx(x,y,z,t)+vy(x,y,z,t)*dvzdy(x,y,z,t)+vz(x,y,z,t)*dvzdz(x,y,z,t)
    return -(rho(scalar)-mu(x,y,z,t)-g)
    

#RK Method For The Function P(x,y,z)
def RKPX(x,y,z,t,P):
    k1=dPdx(x,y,z,t)
    k2=dPdx(x+k1*(h/2),y,z,t+(h/2))
    k3=dPdx(x+k2*(h/2),y,z,t+(h/2))
    k4=dPdx(x+k3,y,z,t+h)
    NewP=P+((h/6)*(k1+2*k2+2*k3+k4))
    return NewP
def RKPY(x,y,z,t,P):
    k1=dPdy(x,y,z,t)
    k2=dPdy(x,y+k1*(h/2),z,t+(h/2))
    k3=dPdy(x,y+k2*(h/2),z,t+(h/2))
    k4=dPdy(x,y+k3,z,t+h)
    NewP=P+((h/6)*(k1+2*k2+2*k3+k4))
    return NewP
def RKPZ(x,y,z,t,P):
    k1=dPdz(x,y,z,t)
    k2=dPdz(x,y,z+k1*(h/2),t+(h/2))
    k3=dPdz(x,y,z+k2*(h/2),t+(h/2))
    k4=dPdz(x,y,z+k3,t+h)
    NewP=P+((h/6)*(k1+2*k2+2*k3+k4))
    return NewP
    




#Update The Grid
def UpdateGrid(t):
    print("Results: ")
    for i in range(20): 
        for j in range(20): 
            for k in range(20):
                        x=i*h
                        y=j*h
                        z=k*h
                        t=0 #Assume t=0

                        grid[i,j,k,0]=RKPX(x,y,z,t,grid[i,j,k,0]) #Pressure Update vx
                        grid[i,j,k,1]=RKPY(x,y,z,t,grid[i,j,k,1]) #Update vy 
                        grid[i,j,k,2]=RKPZ(x,y,z,t,grid[i,j,k,2]) # Update vz

                        if(i==0 or i==19):
                            grid[i,j,k,0]=0 #vx=0
                        if(j==0 or j==19):
                            grid[i,j,k,1]=0 #vy=0
                        if(k==0 or k==19):
                            grid[i,j,k,2]=0 #vz=0
                        print(grid[i,j,k], "\t")
    t+=h #increment the time
    return t #return new time       

                        
def main(): 
    t=0
    for k in range(100):
        t_new=UpdateGrid(t)    

main()  


'''
Excess Code: 
#Runge-Kutta method Partial Derivatives
def RKX(x,y,z,t):
    k1=dvxdt(x,y,z,t)
    k2=dvxdt(x+k1*(h/2),y,z,t+(h/2))
    k3=dvxdt(x+k2*(h/2),y,z,t+(h/2))
    k4=dvxdt(x+k3*h,y,z,t+h)
    return vx(x,y,z,t)+((h/6)*(k1+2*k2+2*k3+k4))

def RKY(x,y,z,t):
    k1=dvydt(x,y,z,t)
    k2=dvydt(x,y+k1*(h/2),z,t+(h/2))
    k3=dvydt(x,y+k2*(h/2),z,t+(h/2))
    k4=dvydt(x,y+k3,z,t+h)
    return vy(x,y,z,t)+((h/6)*(k1+2*k2+2*k3+k4))

def RKZ(x,y,z,t):
    k1=dvzdt(x,y,z,t)
    k2=dvzdt(x,y,z+k1*(h/2),t+(h/2))
    k3=dvzdt(x,y,z+k2*(h/2),t+(h/2))
    k4=dvzdt(x,y,z+k3*(h/2),t+h)
    return vz(x,y,z,t)+((h/6)*(k1+2*k2+2*k3+k4))
'''     