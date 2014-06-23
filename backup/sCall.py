import numpy as np
import matplotlib.pyplot as plt
import swimLibV2 as sl

LH = 424.0
LH = 50
LT = 1503.0
dx = 3.0

#omega = 3.6*2*np.pi
omega = 3.25*2*np.pi
nT = 4.0  # n Periods
dt = 0.001

drivingFunction = 3

E = 3.86
IH = 57.0**3*22.0/12.0
IT = 7.9**3*22.0/12.0
IH = IT
AH = E*IH
AT = E*IT

zetaNHead = 11.19E-9
zetaNTail = 9.25E-9
zetaTHead = zetaNHead/2.0
zetaTTail = zetaNTail/2.0

moment = 37.37
mStart = LH + 45.0   # unit length, not grid number!
mEnd = mStart + 60.0

BC = [1,0]
twist = [0,0]
shift = [0,0]
shiftAmp = 1
twistAmp = 1

params = {'LT':LT, 'LH':LH, 'dx':dx, 'omega':omega, 'nT':nT, 'dt':dt, 'E':E, 'AH':AH, 'AT':AT, 'zetaNHead':zetaNHead, 'zetaNTail':zetaNTail, 'zetaTHead':zetaTHead, 'zetaTTail':zetaTTail, 'moment':moment, 'mStart':mStart, 'mEnd':mEnd, 'BC':BC, 'twist':twist, 'shift':shift, 'shiftAmp':shiftAmp, 'twistAmp':twistAmp, 'drivingFunction':drivingFunction}  


s1 = sl.flagella(params)
s1.actuator()
s1.numSolve()
s1.propulsionCalc()
print(np.mean(s1.Ux))
s1.plotDisp(16,1)




