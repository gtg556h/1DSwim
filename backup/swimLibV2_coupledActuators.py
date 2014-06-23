# Based on swimLibV2, snapshot on 20140611

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
import matplotlib.animation as animation
import random
import pdb

def genWaveform(moment,duration,dt):
    tShort = np.arange(0,duration,dt)
    m = moment * np.sin(tShort*np.pi/duration)
    
    return m


class flagella(object):
    
    def __init__(self, params):
        print('Initializing system')
        self.LT = params['LT']
        self.LH = params['LH']
        self.dx = params['dx']
        self.duration = params['duration']
        self.tMax = params['tMax']
        self.dt = params['dt']
        self.E = params['E']
        self.AH = params['AH']
        self.AT = params['AT']
        self.zetaNHead = params['zetaNHead']
        self.zetaNTail = params['zetaNTail']
        self.zetaTHead = params['zetaTHead']
        self.zetaTTail = params['zetaTTail']
        self.moment = params['moment']
        self.mStart = params['mStart']
        self.mEnd = params['mEnd']
        self.BC = params['BC']
        self.twist = params['twist']
        self.shift = params['shift']
        self.shiftAmp = params['shiftAmp']
        self.twistAmp = params['twistAmp']
        self.nActuators = params['nActuators']
        self.coupling = params['coupling']
        self.randAbs = params['randAbs']
        self.constAbs = params['constAbs']
        self.leak = params['leak']
        self.c0 = params['c0']
        self.actuatorInd = params['actuatorInd']
        self.actuatorCenter = params['actuatorCenter']
        self.surface = params['surface']
        
        self.L = self.LH+self.LT
        self.x = np.arange(0,self.L+self.dx,self.dx)
        self.t = np.arange(0,self.tMax,self.dt)

        self.zetaN = np.zeros(self.x.shape[0])
        self.zetaN[0:np.round(self.LH/self.dx)] = self.zetaNHead
        self.zetaN[np.round(self.LH/self.dx):self.x.shape[0]] = self.zetaNTail
        
        self.zetaT = np.zeros(self.x.shape[0])
        self.zetaT[0:np.round(self.LH/self.dx)] = self.zetaTHead
        self.zetaT[np.round(self.LH/self.dx):self.x.shape[0]] = self.zetaTTail

        self.A = np.zeros(self.x.shape[0])
        self.A[0:np.round(self.LH/self.dx)] = self.AH
        self.A[np.round(self.LH/self.dx):self.x.shape[0]] = self.AT

        self.y0 = np.zeros(self.x.shape[0])
        
        self.c = np.zeros([self.t.shape[0],self.nActuators])
        self.c[0,:] = self.c0
        self.cRandom = np.zeros([self.t.shape[0],self.nActuators])
        self.cConstant = np.zeros([self.t.shape[0],self.nActuators])
        self.cCoupled = np.zeros([self.t.shape[0],self.nActuators])
        self.cLeak = np.zeros([self.t.shape[0],self.nActuators])
        self.state = np.zeros([self.t.shape[0],self.nActuators])
        self.zero = np.zeros([self.t.shape[0], self.nActuators])
        self.phase = np.zeros([self.t.shape[0], self.nActuators])

    ########
    def actuator(self):
        print('Calculating normal driving forces w(x,t)')

        mFunc = genWaveform(self.moment, self.duration, self.dt)

        self.m = np.zeros([self.x.shape[0],self.t.shape[0]])
        self.w = np.zeros([self.x.shape[0],self.t.shape[0]])

        self.mFunc = mFunc

        #self.momentCalc(mFunc)


    #######

    def momentCalc(self,m):
        m[0:2] = np.zeros([2])
        m[m.shape[0]-2:m.shape[0]] = np.zeros([2])

        a = np.zeros([m.shape[0],m.shape[0]])
        #w = np.zeros(self.x.shape[0])

        for i in range(0, self.x.shape[0]-1):
            a[i,i+1:m.shape[0]] = self.dx*np.arange(1,m.shape[0]-i,1)

        a[m.shape[0]-1,:] = np.ones([1,m.shape[0]])

        w = np.linalg.solve(a,m)

        return w

        #return self.w

    ###########################
    # Function to step actuator state:

    def stepActuator(self,i):
        # utilize self.c and self.state
        # upper bound on state: length of mFunc:
        thresh = self.mFunc.shape[0]-1

        bendingVec = self.calcBending(self.y[:,i-1],self.dx)
        
        for jj in range(0,self.nActuators):
            if self.state[i-1,jj]>0:
                #print('jj',jj)
                #print('i',i)
                #pdb.set_trace()
                self.state[i,jj] = self.state[i-1,jj]+1
                self.m[self.actuatorInd[0,jj]:self.actuatorInd[1,jj],i] = self.mFunc[self.state[i-1,jj]] * self.surface[jj]
                #self.state[i,jj] = self.state[i-1,jj]+1
                if self.state[i,jj] > thresh:
                    self.state[i,jj] = 0
            else:
                self.cCoupled[i,jj] = self.dt*np.max([-self.coupling[jj]*bendingVec[self.actuatorCenter[jj]]*self.surface[jj],0])
                #pdb.set_trace()
               
                self.cConstant[i,jj] = self.dt*self.constAbs[jj]
                self.cRandom[i,jj] = self.dt*self.randAbs[jj]*random.random()
                self.cLeak[i,jj] = self.dt*self.leak[jj]*self.c[i-1,jj]

                #self.c[i,jj] = self.c[i-1,jj] + self.constAbs[jj] + self.coupling[jj]*bendingVec[self.actuatorCenter[jj]]
                self.c[i,jj] = self.c[i-1,jj] + self.cCoupled[i,jj] + self.cConstant[i,jj] + self.cRandom[i,jj] - self.cLeak[i,jj]
                if self.c[i,jj] > 1:
                    self.state[i,jj]=1
                    self.c[i,jj] = 0

        self.w[:,i] = self.momentCalc(self.m[:,i])
        #if (self.state[i,:] != np.array([0,0])).any():
            #pdb.set_trace()



    #######

    def calcBending(self,y,dx):
        A = np.zeros([y.shape[0],y.shape[0]])
        for ii in range(1,y.shape[0]-1):
            A[ii,ii-1:ii+2] = [1,-2,1]
        A[0,0:4] = [2,-5,4,-1]
        A[y.shape[0]-1,y.shape[0]-4:y.shape[0]] = [-1,4,-5,2]

        A = A/dx**2

        return np.dot(A,y)
        

    #######
    def numSolve(self):
        print('Solving for y(x,t)')
        nx = self.x.shape[0]
        nt = self.t.shape[0]

        self.y = np.zeros([nx,nt])
        self.y[:,0] = self.y0

        D4 = np.zeros([nx,nx])
        Dt = np.zeros([nx,nx])
        a = np.zeros([nx,nx])
        c = np.zeros([nx,nt])

        for i in range(2,nx-2):
            D4[i,i-2:i+3] = self.A[i]*np.array([1,-4,6,-4,1])/self.dx**3

        Dt[2:nx-2,2:nx-2] = np.diag(self.zetaN[2:nx-2]*self.dx/self.dt)

        # LH BC
        if self.BC[0]==1:
            a[0,0] = 1
            a[1,1] = 1

            if self.shift[0]==1:
                c[0,:] = self.shiftAmp*np.sin(self.omega*self.t)
            else:
                c[0,:] = np.zeros(nt)

            if self.twist[0] == 1:
                c[1,:] = c[1,:] + self.dx*self.twistAmp*np.sin(self.omega*self.t)
            else:
                c[1,:] = c[0,:]

        else:
            a[0,0:4] = np.array([2,-5,4,-1])/self.dx**2
            a[1,0:4] = np.array([-1,3,-3,1])/self.dx**3
            c[0:2,:] = np.zeros([2,nt])

        # RH BC
        if self.BC[1]==1:
            a[nx-2,nx-2] = 1
            a[nx-1,nx-1] = 1

            if self.shift[1] == 1:
                c[nx-1,:] = self.shiftAmp*np.sin(self.omega*self.t)
            else:
                c[nx-1,:] = np.zeros(self.t.shape[0])

            if self.twist[1] == 1:
                c[nx-2,:] = c[nx-1,:] - self.dx*self.twistAmp*np.sin(self.omega*self.t)
            else:
                c[nx-2,:] = c[nx-1,:]

        else:
            a[nx-2,nx-4:nx] = np.array([-1,3,-3,1])/self.dx**3
            a[nx-1,nx-4:nx] = np.array([-1,4,-5,2])/self.dx**2 
            c[nx-2:nx,:] = np.zeros([2,nt])

        # Old generation of c, not useful now...
        #c[2:nx-2,:] = c[2:nx-2,:] + self.w[2:nx-2,:]
        
        # Build differential operator
        a = a + D4 + Dt
        
        # Solution step
        for i in range(1,nt):
            #print(i,'/',nt)
            if np.mod(i,50)==0:
                print(i/np.float(self.t.shape[0]))


            # Update actuator, calculate contractile force:
            self.stepActuator(i)

            # Add current moment from cell contraction:
            c[2:nx-2,i] = c[2:nx-2,i] + self.w[2:nx-2,i]
                
            # Add other stuff from previous step, etc...
            c[2:nx-2,i] = c[2:nx-2,i] + np.multiply(self.zetaN[2:nx-2],self.y[2:nx-2,i-1])*self.dx/self.dt

            self.y[:,i] = np.linalg.solve(a,c[:,i])


    #######
    def genPhase(self):
        self.zero[np.where(self.state == np.max(self.state))] = 1

        for i in range(0, self.nActuators):
            zeroIndex = np.where(self.state[:,i] == np.max(self.state))[0]

            meanPeriod = np.mean(np.diff(zeroIndex)[1:zeroIndex.shape[0]-1])
            print(meanPeriod)

            for k in range(0, zeroIndex[0]+1):
                self.phase[k,i] = ((meanPeriod - zeroIndex[0]) + k)/(meanPeriod-1)

            for j in range(0, zeroIndex.shape[0]-1):
                for k in range(zeroIndex[j], zeroIndex[j+1]):
                    self.phase[k,i] = (k-zeroIndex[j])/(zeroIndex[j+1] - zeroIndex[j] - 1)
                    #pdb.set_trace()
                    #print(i,j,k)

            for k in range(zeroIndex[-1], self.phase.shape[0]):
                self.phase[k,i] = (k - zeroIndex[-1])/meanPeriod



    #######
    def genPhaseDiff(self,ref):
        self.phaseDiff = np.zeros(self.phase.shape)
        
        for i in range(0,self.nActuators):
            self.phaseDiff[:,i] = np.mod((self.phase[:,i] - self.phase[:,ref]),1)

    #######
    def propulsionCalc(self):
        print('Post-processing data')
        zetaP = self.zetaN-self.zetaT

        nx = self.x.shape[0]
        nt = self.t.shape[0]

        DX = np.zeros([nx,nx])
        self.X = np.zeros([nx,nt])
        self.XRef = np.zeros(nt)
        self.Ux = np.zeros(nt)
        self.prop = np.zeros(nt)

        for i in range(1,nx-1):
            DX[i,i-1:i+2] = np.array([-1,0,1])/(2*self.dx)

        for i in range(1,self.t.shape[0]):
            yx = np.dot(DX,self.y[:,i])
            yt = (self.y[:,i]-self.y[:,i-1])/self.dt
            self.prop[i] = np.sum(np.multiply(np.multiply(yt,yx),zetaP)*self.dx)

            self.Ux[i] = self.prop[i]/(np.sum(self.zetaT*self.dx))
            self.XRef[i] = self.XRef[i-1] + self.Ux[i]*self.dt
            self.X[:,i] = self.x+self.XRef[i]


    #######

    def plotDisp(self,DF=1,plotFrac=1):
        print('Displaying solution shapes')
        nx = self.x.shape[0]
        nt = self.t.shape[0]

        nFrames = np.int(np.floor(nt/DF*plotFrac))
        #print(nFrames)
        yRange = np.max(self.y)-np.min(self.y)
        xRange = np.max(self.X)-np.min(self.X)

        figScale = 8.0
        fig = plt.figure(figsize=[figScale,yRange/np.max(self.x)*figScale])
        ax = plt.axes()

        ax.set_xlim([np.min(self.X)-.1*xRange,np.max(self.X)+.1*xRange])
        ax.set_ylim([np.min(self.y)-.25*yRange,np.max(self.y)+.25*yRange])
        ax.set_yticklabels([])
        
        line, = ax.plot([], [], lw=2)

        # Initialization function: plot the background of each frame
        def init():
            line.set_data([], [])
            return line,

        # animation function, called sequentially:
        def animate(i):
            y2 = self.y[:,DF*i]
            X2 = self.X[:,DF*i]
            #line.set_data(self.x,y2)
            line.set_data(X2,y2)
            return line,

        # Call the animator:
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nFrames, interval=50, blit=True, repeat=False)

        plt.show()


   
        

########################################
########################################
########################################

def plotSave(t,x,y,fileDir,prefix):
    #curDir = os.getcwd()
    #os.chdir(fileDir)
    
    fig = plt.figure(figsize=(4,2))
    ax1 = fig.add_subplot(111)
    ax1.hold(False)

    nFrames = t.shape[0]
    nZeros = np.ceil(np.log10(nFrames)) + 1
    frameNumber = 0
    stem = fileDir+prefix

    for ii in range(0,nFrames,40):
        ax1.plot(x,y[:,ii])

        frameNumber = frameNumber + 1
        filename = stem + "%06d" % (frameNumber,) 
        filename += '.png'
        plt.savefig(filename, dpi=160, facecolor='w')

    #os.chdir(curDir)

########################################
########################################
########################################








