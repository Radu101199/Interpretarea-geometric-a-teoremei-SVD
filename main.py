import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('macosx')

m = int(input('Input the m dimension of A:\n'))
n = int(input('Input the n dimension of A:\n'))

A = np.random.rand(m, n)

def get_matrix_rank(A):
    return la.matrix_rank(A)

def svd(A):
    return la.svd(A)

def build_sigma(A, s):
    S = np.zeros_like(A)
    r = get_matrix_rank(A)
    S[:r, :r] = np.diag(s[:r])
    
    return S

def verify_svd(A):
    U, s, Vt = svd(A)
    S = build_sigma(A, s)
    
    return np.allclose(A, np.dot(np.dot(U, S), Vt))

def plot_circle2D():
    theta = np.linspace(0, 2 * np.pi, 1000)
    x, y = np.cos(theta), np.sin(theta) # Desenarea cercului
    
    return x, y
    
def plot_ellipsis2D(minor_axis, major_axis):
    theta = np.linspace(0, 2 * np.pi, 100)
    x, y = major_axis * np.cos(theta), minor_axis * np.sin(theta)
    
    plt.axis('equal')
    plt.plot(x, y)
    plt.show()
    
def plot_transformed_circle(A):
    theta = np.linspace(0, 2 * np.pi, 1000)
    x, y = np.cos(theta), np.sin(theta)
    
    x_transf, y_transf = np.zeros_like(x), np.zeros_like(y)
    
    for i in range(len(theta)):
        p = [x[i], y[i]]
        p_transf = np.dot(A, p)
        x_transf[i], y_transf[i] = p_transf[0], p_transf[1]
    
    return x_transf, y_transf   

def plot_sphere(units):
    t1 = np.linspace(-np.pi / 2, np.pi / 2, units)
    t2 = np.linspace(0, 2 * np.pi, units)
    
    x = np.outer(np.cos(t1), np.cos(t2)) # Desenarea sferei
    y = np.outer(np.cos(t1), np.sin(t2))
    z = np.outer(np.sin(t1), np.ones(units))
    
    return x, y, z

def plot_transformed_sphere(A, units):
    t1 = np.linspace(-np.pi / 2, np.pi / 2, units)
    t2 = np.linspace(0, 2 * np.pi, units)
    
    x = np.outer(np.cos(t1), np.cos(t2))
    y = np.outer(np.cos(t1), np.sin(t2))
    z = np.outer(np.sin(t1), np.ones(units))
    
    x_transf, y_transf, z_transf = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
    
    for i in range(len(t1)):
        for j in range(len(t2)):
            p = [x[i, j], y[i, j], z[i, j]]
            p_transf = np.dot(A, p)
            
            x_transf[i, j], y_transf[i, j], z_transf[i, j] = p_transf[0], p_transf[1], p_transf[2]

    return x_transf, y_transf, z_transf
        
def solve2D(A):
    U, s, Vt = svd(A)
    S = build_sigma(A, s)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    x, y = plot_circle2D()
    
    ax1.set_title('Cercul unitate')
    ax1.plot(x, y)
    
    ax1.quiver(0, 0, Vt[0, 0], Vt[1, 0], angles='xy', scale_units='xy', scale=1)
    ax1.text(Vt[0, 0] / 2, Vt[1, 0] / 2, 'v1', fontsize=12)
    
    ax1.quiver(0, 0, Vt[0, 1], Vt[1, 1], angles='xy', scale_units='xy', scale=1)
    ax1.text(Vt[0, 1] / 2, Vt[1, 1] / 2, 'v2', fontsize=12)
    
    ax1.axis('equal')
    
    x, y = plot_transformed_circle(A)
    su = np.dot(U, S)
    
    ax2.set_title('Cercul unitate dupa transformare')
    ax2.plot(x, y)
    
    ax2.quiver(0, 0, su[0, 0], su[1, 0], angles='xy', scale_units='xy', scale=1)
    ax2.text(su[0, 0] / 2, su[1, 0] / 2, 's1u1', fontsize=12)
    
    ax2.quiver(0, 0, su[0, 1], su[1, 1], angles='xy', scale_units='xy', scale=1)
    ax2.text(su[0, 1] / 2, su[1, 1] / 2, 's2u2', fontsize=12)
    
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
def solve3D(A):
    U, s, Vt = svd(A)
    S = build_sigma(A, s)
    
    fig = plt.figure(figsize=(10, 5))
    
    x, y, z = plot_sphere(100)
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Sfera unitate')
    ax1.plot_surface(x, y, z, alpha=0.3)
    
    ax1.quiver(0, 0, 0, Vt[0, 0], Vt[1, 0], Vt[2, 0])
    ax1.text(Vt[0, 0] / 2, Vt[1, 0] / 2, Vt[2, 0] / 2, 'v1', fontsize=12)
    
    ax1.quiver(0, 0, 0, Vt[0, 1], Vt[1, 1], Vt[2, 1])
    ax1.text(Vt[0, 1] / 2, Vt[1, 1] / 2, Vt[2, 1] / 2, 'v2', fontsize=12)
    
    ax1.quiver(0, 0, 0, Vt[0, 2], Vt[1, 2], Vt[2, 2])
    ax1.text(Vt[0, 2] / 2, Vt[1, 2] / 2, Vt[2, 2] / 2, 'v3', fontsize=12)
    
    ax1.axis('auto')
    
    x, y, z = plot_transformed_sphere(A, 100)
    su = np.dot(U, S)
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Sfera unitate dupa transformare')
    ax2.plot_surface(x, y, z, alpha=0.3)
    
    ax2.quiver(0, 0, 0, su[0, 0], su[1, 0], su[2, 0])
    ax2.text(su[0, 0] / 2, su[1, 0] / 2, su[2, 0] / 2, 's1u1', fontsize=12)
    
    ax2.quiver(0, 0, 0, su[0, 1], su[1, 1], su[2, 1])
    ax2.text(su[0, 1] / 2, su[1, 1] / 2, su[2, 1] / 2, 's2u2', fontsize=12)
    
    ax2.quiver(0, 0, 0, su[0, 2], su[1, 2], su[2, 2])
    ax2.text(su[0, 2] / 2, su[1, 2] / 2, su[2, 2] / 2, 's3u3', fontsize=12)
    
    ax2.axis('auto')

    ax2.view_init(50, -20)
    
    plt.show()
   
# solve2D(A)

# B = A.copy()
# B[1] = B[0]
#
# solve2D(B)

solve3D(A)

# B = A.copy()
# B[0] = B[1]
# #
# B[0] = B[1]
# B[2] = B[0]
# #
# solve3D(B)