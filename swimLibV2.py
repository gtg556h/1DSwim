import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
import matplotlib.animation as animation

def genWaveform1(t,omega):
    # Standard waveform used in NatureComm paper

    m = np.sin(omega/2.0*t)
    m = (np.abs(m) - 0.3)/0.7
    m[np.where(m<0)[0]] = 0

    return m

def genWaveform2(t,omega):
    # Pefect sinusoidal, 2-sided driving function

    m = np.sin(omega*t)
    #m = (np.abs(m) - 0.3)/0.7
    #m[np.where(m<0)[0]] = 0

    return m

def genWaveform3(t,omega):
    # Nominally tweaked waveform form conference proceedings etc...

    m = np.sin(omega/2.0*t)
    m = (np.abs(m) - 0.25)/0.75
    m[np.where(m<0)[0]] = 0

    return m



class flagella(object):
    
    def __init__(self, params):
        print('Initializing system')
        self.LT = params['LT']
        self.LH = params['LH']
        self.dx = params['dx']
        self.omega = params['omega']
        self.nT = params['nT']
        self.dt = params['dt']
        self.E = params['E']
        self.AH = params['AH']
        self.AT = params['AT']
        self.zetaNHead = params['zetaNHead']
        self.zetaNTail = params['zetaNTail']
        self.zetaTHead = params['zetaTHead']
        self.zetaTTail = params['zetaTTail']
        self.drivingFunction = params['drivingFunction']
        self.moment = params['moment']
        self.mStart = params['mStart']
        self.mEnd = params['mEnd']
        self.BC = params['BC']
        self.twist = params['twist']
        self.shift = params['shift']
        self.shiftAmp = params['shiftAmp']
        self.twistAmp = params['twistAmp']
        self.drivingFunction = params['drivingFunction']
        
        self.L = self.LH+self.LT
        self.x = np.arange(0,self.L+self.dx,self.dx)
        self.T = 2*np.pi/self.omega 
        self.t = np.arange(0,self.nT*self.T+self.dt,self.dt)

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

    ########
    def actuator(self):
        print('Calculating normal driving forces w(x,t)')
        if self.drivingFunction == 2:
            mFunc = genWaveform2(self.t,self.omega)
        elif self.drivingFunction == 3:
            mFunc = genWaveform3(self.t,self.omega)
        else:
            mFunc = genWaveform1(self.t,self.omega)

        #mFunc = genWaveform1(self.t,self.omega)
        self.m = np.zeros(self.x.shape[0])
        mStartIndex = np.round(self.mStart/self.dx)
        mEndIndex = np.round(self.mEnd/self.dx)
        self.m[mStartIndex:mEndIndex+1] = self.moment

        self.mFunc = mFunc

        self.momentCalc(mFunc)


    #######

    def momentCalc(self,mFunc):
        self.m[0:2] = np.zeros([2])
        self.m[self.m.shape[0]-2:self.m.shape[0]] = np.zeros([2])

        a = np.zeros([self.m.shape[0],self.m.shape[0]])
        self.w = np.zeros([self.x.shape[0],self.t.shape[0]])

        for i in range(0, self.x.shape[0]-1):
            a[i,i+1:self.m.shape[0]] = self.dx*np.arange(1,self.m.shape[0]-i,1)

        a[self.m.shape[0]-1,:] = np.ones([1,self.m.shape[0]])

        wBase = np.linalg.solve(a,self.m)

        for i in range(0,self.t.shape[0]):
            self.w[:,i] = np.multiply(wBase,mFunc[i])

        #return self.w


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

        c[2:nx-2,:] = c[2:nx-2,:] + self.w[2:nx-2,:]
        
        # Build differential operator
        a = a + D4 + Dt
        
        # Solution step
        for i in range(1,nt):
            if np.mod(i,50)==0:
                print(i/np.float(self.t.shape[0]))
                
            c[2:nx-2,i] = c[2:nx-2,i] + np.multiply(self.zetaN[2:nx-2],self.y[2:nx-2,i-1])*self.dx/self.dt

            self.y[:,i] = np.linalg.solve(a,c[:,i])


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
		# mac issues: 
		# Error: 'FigureCanvasMac' object has no attribute 'restore_region'
		# Solution:
		# 1. blit=False
		#    ani = animation.FuncAnimation(fig, ...., blit=False)
		# 2. import matplotlib
	    #    matplotlib.use('TkAgg')
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
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nFrames, interval=50, blit=False, repeat=True)

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








