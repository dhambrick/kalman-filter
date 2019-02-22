import numpy as np 
import modelgenerator as mg 
import unscentedtransform as ut

def radarSensorMeasurement(state):
    radarPos = {"xr":0,"yr":0}
    xdiff = (state[0]-radarPos["xr"])
    ydiff = (state[1]-radarPos["yr"])
    r = np.sqrt(xdiff*xdiff + ydiff*ydiff)
    theta = np.arctan2(ydiff/xdiff)
    return np.array([r,theta]) 
