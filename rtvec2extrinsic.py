import os
import math
import numpy as np

def rtvec2matrix(rw, rx, ry, rz, tx, ty, tz):

    T = np.array([[
        1 - 2*ry**2 - 2*rz**2,
        2*rx*ry - 2*rz*rw,
        2*rx*rz + 2*ry*rw,
        tx],
    
        [2*rx*ry + 2*rz*rw,
        1 - 2*rx**2 - 2*rz**2,
        2*ry*rz - 2*rx*rw,
        ty],
    
        [2*rx*rz - 2*ry*rw,
        2*ry*rz + 2*rx*rw,
        1 - 2*rx**2 - 2*ry**2,
        tz],
    
        [0, 0, 0, 1]])
    return np.array(T)


def eulerAngles2rotationMat(theta, format='degree'):
    if format == 'degree':
        theta = [i * math.pi / 180.0 for i in theta]

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
 
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
 
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def angelPos2Transformation(rx, ry, rz, tx, ty, tz):
    R = eulerAngles2rotationMat([rx, ry, rz])
    T = np.c_[R, [tx, ty, tz]]
    T = np.r_[T, [[0, 0, 0, 1]]]
    return T

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return x, y, z

def transformation2AnglePos(T):
    [tx, ty, tz, _] = T[:,3]
    R = np.array(T[:3,:3])
    rx, ry, rz = rotationMatrixToEulerAngles(R)
    return rx, ry, rz, tx, ty, tz