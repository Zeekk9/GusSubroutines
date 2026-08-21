import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def Seidel(x, y, a, b, c, d, e, f, g):
    s = a*((x**2)+(y**2))**2+b*((x**2)+(y**2))*(y)+c * \
        ((x**2)+(3*(y**2)))+d * \
        ((x**2)+(y**2))+(e*(y))+(f*(x))+g
    return s


def csc(sigma):
    csc = 1/np.sin(sigma)
    return csc


def sec(sigma):
    sec = 1/np.cos(sigma)
    return sec


def cot(sigma):
    cot = np.cos(sigma)/np.sin(sigma)
    return cot


def c_matx(n, m, error, shape):
    c = np.zeros(shape)+1/4*np.sinc(1/2*n)*np.sinc(1/2*m)
    c = np.random.normal(c, c*error, np.shape(shape))
    return c


def c1_matx(n, m, error, shape):
    c = np.zeros(shape)+1/4*np.sinc(1/2*(n+1))*np.sinc(1/2*m)
    c = np.random.normal(c, c*error, shape)
    return c


def c(n, m):
    c = 0.5*np.sinc(0.5*n)*0.5*np.sinc(0.5*m)
    return c


def Inm_All(Ar, Ap, phi1, phi2, theta, ep_theta, sigma_m, cnm, cn1m):
    I = np.conj(Ar*np.exp(1j*phi1)*cnm*np.cos(sigma_m)**2+Ap*np.exp(1j*phi2)*cn1m*((1-1j)*np.cos(ep_theta+theta)*np.cos(sigma_m)*np.sin(ep_theta+theta)*np.sin(sigma_m)+np.cos(sigma_m)**2*(np.cos(ep_theta+theta)**2+1j*np.sin(ep_theta+theta)**2)))*(Ar*np.exp(1j*phi1)*cnm*np.cos(sigma_m)**2+Ap*np.exp(1j*phi2)*cn1m*((1-1j)*np.cos(ep_theta+theta)*np.cos(sigma_m)*np.sin(ep_theta+theta)*np.sin(sigma_m)+np.cos(sigma_m)**2*(np.cos(ep_theta+theta)**2+1j*np.sin(ep_theta+theta)**2)))+np.conj(Ar *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     np.exp(1j*phi1)*cnm*np.cos(sigma_m)*np.sin(sigma_m)+Ap*np.exp(1j*phi2)*cn1m*(np.cos(sigma_m)*np.sin(sigma_m)*(np.cos(ep_theta+theta)**2+1j*np.sin(ep_theta+theta)**2)+(1-1j)*np.cos(ep_theta+theta)*np.sin(ep_theta+theta)*np.sin(sigma_m)**2))*(Ar*np.exp(1j*phi1)*cnm*np.cos(sigma_m)*np.sin(sigma_m)+Ap*np.exp(1j*phi2)*cn1m*(np.cos(sigma_m)*np.sin(sigma_m)*(np.cos(ep_theta+theta)**2+1j*np.sin(ep_theta+theta)**2)+(1-1j)*np.cos(ep_theta+theta)*np.sin(ep_theta+theta)*np.sin(sigma_m)**2))
    return I


def Inm_All_s(Ar, Ap, phi1, phi2, theta, ep_theta, cnm, cn1m):
    I = 1/2*np.exp(2*1j*(phi2-phi2.real))*(np.abs(Ap*cn1m))**2*np.sin(2*(ep_theta+theta))*np.sin(2*np.conj(ep_theta+theta))+np.conj(Ar*np.exp(1j*phi1)*cnm+Ap*np.exp(1j*phi2)
                                                                                                                                    * cn1m*(np.cos(ep_theta+theta)**2+1j*np.sin(ep_theta+theta)**2))*(Ar*np.exp(1j*phi1)*cnm+Ap*np.exp(1j*phi2)*cn1m*(np.cos(ep_theta+theta)**2+1j*np.sin(ep_theta+theta)**2))
    return I


def Inm_u(u0, u1, u2, u3, sigma):
    Im = (u0+u1+u2)+(u0+u2)*np.cos(2*sigma)+u3*np.sin(2*sigma)
    return Im


def Inm_u_s(u0, u1, u2):
    Im = (u0+2*u1+u2)+(u0+u2)
    return Im


def Inm(Ar, Ap, phi, n, m, sigma):
    I2m = c(n, m)**2*np.cos(sigma)**2*Ar**2+1/2*c(n+1, m)**2*Ap**2+np.sqrt(2) * \
        c(n, m)*c(n+1, m)*np.cos(sigma)*Ar*Ap*np.cos(phi+sigma)
    return I2m


def Inm_s(Ar, Ap, phi, n, m):
    I2m = c(n, m)**2*Ar**2+c(n+1, m)**2*Ap**2+np.sqrt(2) * \
        c(n, m)*c(n+1, m)*Ar*Ap*np.cos(phi)
    return I2m


def gradtorad(x):
    r = x*np.pi/180
    return r


def itoh_2D(W):
    renglon, columna = W.shape
    phi = np.zeros(W.shape)
    psi = np.zeros(W.shape)
    phi[0, 0] = W[0, 0]
    # Se Desenvuelve la primera columna
    for m in range(1, columna):
        Delta = W[0, m] - W[0, m - 1]
        WDelta = np.arctan2(np.sin(Delta), np.cos(Delta))
        phi[0, m] = phi[0, m - 1] + WDelta
    psi[0, :] = phi[0, :]

    for k in range(columna):
        psi[0, k] = W[0, k]
        for p in range(1, renglon):
            Delta = W[p, k] - W[p - 1, k]
            WDelta = np.arctan2(np.sin(Delta), np.cos(Delta))
            phi[p, k] = phi[p - 1, k] + WDelta
    return phi


def matshow(position, mat, title):
    plt.subplot(position)
    plt.imshow(mat, cmap='gist_heat')
    plt.title(title, fontsize=30)
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)


def Plot3D(position, W, title):

    xlim = W[0, :].size
    ylim = W[:, 0].size
    x = np.linspace(0, xlim, xlim)
    y = np.linspace(0, ylim, ylim)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    fig.suptitle(title, fontsize=40)
    ax = fig.add_subplot(position, projection='3d')
    ax.plot_surface(X, Y, W, cmap='gist_heat')
    plt.axis('on')
    plt.grid()


def prom_filter(M):
    x, y = M.shape
    M_filter = np.zeros((x-1, y-1))
    Sum = 0
    # print(M.shape)
    for i in range(x-2):
        # print('cambio2')
        for j in range(y-2):
            # print('cambio')
            Sum = 0
            for k in range(3):
                for l in range(3):
                    #print('posicion:', i+k, j+k, ' Valor:', M[i+k, j+l])
                    Sum += M[i+k, j+l]
                    if k == 2 and l == 2:
                        M_filter[i+1, j+1] = Sum/9
                        # print(M[i+1,j+1])
                        # print(i+1,j+1)
    return M_filter[1:x-1, 1:y-1]


def P_10(I00, I_10, I01, I_11, sigma):
    # -1,0
    u0 = (((c(0, 0)**3-c(0, 0)*c(1, 0)**2)*(I01*c(1, 0)**2-I_11*c(1, 1)**2)+sec(2*sigma)*c(0, 0)*(c(1, 0)**2+c(1, 1)**2)*((-I_11+I01) *
          c(0, 0)**2+(2*I_10+I_11-2*I00-I01)*c(1, 0)**2+2*(-I_10+I00)*c(1, 1)**2)))/((2*(c(0, 0)**2-c(1, 0)**2)*c(1, 1)*(c(1, 0)**2-c(1, 1)**2)))

    u1 = 1/(2*(c(0, 0)**2-c(1, 0)**2)*(c(1, 0)**2-c(1, 1)**2))*((1+sec(2*sigma))*(I_11-I01)*c(0, 0)**4+sec(2*sigma)*c(0, 0)
                                                                ** 2*(-((2*I_10+I_11+np.cos(2*sigma)*I_11-2*I00-2*(np.cos(sigma))**2*I01)*c(1, 0)**2)+2*(I_10-I00)*c(1, 1)**2))

    u2 = (sec(2*sigma)*c(1, 0)**2*((I_11-I01)*c(0, 0)**2+(-2*I_10-I_11+2*I00+I01)*c(1, 0)
          ** 2+2*(I_10-I00)*c(1, 1)**2))/(2*(c(0, 0)**2-c(1, 0)**2)*(c(1, 0)**2-c(1, 1)**2))

    u3 = ((csc(2*sigma)*(2*(np.cos(sigma))**2*sec(2*sigma)*c(0, 0)*c(1, 0)**2*((I_11-2*(np.cos(sigma))**2*I01)*c(0, 0)**2-(2*I_10+I_11-2*I00-2*(np.cos(sigma))**2*I01)*c(1, 0)**2)+(-((1+sec(2*sigma))*(I_11-I01)*c(0, 0)**4)+2*sec(2*sigma)*(2*(np.cos(sigma))**2*I_10-I00)*c(0, 0)**2*c(1, 0)**2+(I_11-2*I00+sec(2*sigma)*(2*I_10+I_11-2*I00-I01)-I01)*c(1, 0)**4)
          * c(1, 1)+2*(np.cos(sigma))**2*sec(2*sigma)*(2*(np.cos(sigma))**2*I_11-I01)*c(0, 0)*(c(0, 0)**2-c(1, 0)**2)*c(1, 1)**2+2*sec(2*sigma)*((-2*(np.cos(sigma))**2*I_10+I00)*c(0, 0)**2-(I_10-2*(np.cos(sigma))**2*I00)*c(1, 0)**2)*c(1, 1)**3+2*(1+sec(2*sigma))*(I_10-I00)*c(0, 0)*c(1, 1)**4)))/((2*(c(0, 0)**2-c(1, 0)**2)*c(1, 1)*(c(1, 0)**2-c(1, 1)**2)))

    Ap = 2*u1/c(1, 0)**2
    Ar = 2*u2/c(0, 0)**2
    phi = np.arctan2(u3, u0)
    phi_ = np.nan_to_num(np.arctan2(u1-np.nan_to_num(np.sqrt((u0+u1+u2) **
                                                             2-(u0+u2)**2-u3**2)), u3))

    return Ap, Ar, phi, phi_, u0, u1, u2, u3


def P00(I00, I_10, I01, I_11, sigma):

    # 0,0
    u0 = (((c(0, 0)**3-c(0, 0)*c(1, 0)**2)*(I01*c(1, 0)**2-I_11*c(1, 1)**2)+sec(2*sigma)*c(0, 0)*(c(1, 0)**2+c(1, 1)**2)*((-I_11+I01) *
          c(0, 0)**2+(2*I_10+I_11-2*I00-I01)*c(1, 0)**2+2*(-I_10+I00)*c(1, 1)**2)))/((2*(c(0, 0)**2-c(1, 0)**2)*c(1, 1)*(c(1, 0)**2-c(1, 1)**2)))

    u1 = ((c(1, 0)**2*(-((1+sec(2*sigma))*(I_11-I01)*c(0, 0)**2)+sec(2*sigma)*((2*I_10+I_11+np.cos(2*sigma)*I_11-2*I00-2 *
          (np.cos(sigma))**2*I01)*c(1, 0)**2+2*(-I_10+I00)*c(1, 1)**2))))/((2*(-c(0, 0)**2+c(1, 0)**2)*(c(1, 0)**2-c(1, 1)**2)))

    u2 = (sec(2*sigma)*c(0, 0)**2*((I_11-I01)*c(0, 0)**2+(-2*I_10-I_11+2*I00+I01)*c(1, 0)
          ** 2+2*(I_10-I00)*c(1, 1)**2))/(2*(c(0, 0)**2-c(1, 0)**2)*(c(1, 0)**2-c(1, 1)**2))

    u3 = ((csc(2*sigma)*(2*(np.cos(sigma))**2*sec(2*sigma)*c(0, 0)*c(1, 0)**2*((I_11-2*(np.cos(sigma))**2*I01)*c(0, 0)**2-(2*I_10+I_11-2*I00-2*(np.cos(sigma))**2*I01)*c(1, 0)**2)+(-((1+sec(2*sigma))*(I_11-I01)*c(0, 0)**4)+2*sec(2*sigma)*(2*(np.cos(sigma))**2*I_10-I00)*c(0, 0)**2*c(1, 0)**2+(I_11-2*I00+sec(2*sigma)*(2*I_10+I_11-2*I00-I01)-I01)*c(1, 0)**4)
          * c(1, 1)+2*(np.cos(sigma))**2*sec(2*sigma)*(2*(np.cos(sigma))**2*I_11-I01)*c(0, 0)*(c(0, 0)**2-c(1, 0)**2)*c(1, 1)**2+2*sec(2*sigma)*((-2*(np.cos(sigma))**2*I_10+I00)*c(0, 0)**2-(I_10-2*(np.cos(sigma))**2*I00)*c(1, 0)**2)*c(1, 1)**3+2*(1+sec(2*sigma))*(I_10-I00)*c(0, 0)*c(1, 1)**4)))/((2*(c(0, 0)**2-c(1, 0)**2)*c(1, 1)*(c(1, 0)**2-c(1, 1)**2)))

    Ap = 2*u1/c(1, 0)**2
    Ar = 2*u2/c(0, 0)**2
    #Ar = (2*(u0+u1+u2)-2*np.sqrt((u0+u1+u2)**2-(u0+u2)**2-u3**2))/c(0, 0)
    phi = np.arctan2(u3, u0)
    phi_ = np.arctan2(u1-np.sqrt((u0+u1+u2)**2-(u0+u2)**2-u3**2), u3)

    return Ap, Ar, phi, phi_, u0, u1, u2, u3


def P01(I00, I_10, I01, I_11, sigma):

    # 0,1
    u0 = 1/(2*(c(0, 0)**2-c(1, 0)**2)*(c(1, 0)**2-c(1, 1)**2))*((c(0, 0)**2-c(1, 0)**2)*(I01*c(1, 0)**2-I_11*c(1, 1)**2) +
                                                                sec(2*sigma)*(c(1, 0)**2+c(1, 1)**2)*((-I_11+I01)*c(0, 0)**2+(2*I_10+I_11-2*I00-I01)*c(1, 0)**2+2*(-I_10+I00)*c(1, 1)**2))

    u1 = 1/(2*(c(0, 0)**2-c(1, 0)**2)*(c(1, 0)**2-c(1, 1)**2))*c(1, 1)**2*((1+sec(2*sigma))*(I_11-I01)*c(0, 0)**2 +
                                                                           sec(2*sigma)*(-((2*I_10+I_11+np.cos(2*sigma)*I_11-2*I00-2*(np.cos(sigma))**2*I01)*c(1, 0)**2)+2*(I_10-I00)*c(1, 1)**2))

    u2 = (sec(2*sigma)*c(1, 0)**2*((I_11-I01)*c(0, 0)**2+(-2*I_10-I_11+2*I00+I01)*c(1, 0)
          ** 2+2*(I_10-I00)*c(1, 1)**2))/(2*(c(0, 0)**2-c(1, 0)**2)*(c(1, 0)**2-c(1, 1)**2))

    u3 = ((csc(2*sigma)*(2*(np.cos(sigma))**2*sec(2*sigma)*c(0, 0)*c(1, 0)**2*((I_11-2*(np.cos(sigma))**2*I01)*c(0, 0)**2-(2*I_10+I_11-2*I00-2*(np.cos(sigma))**2*I01)*c(1, 0)**2)+(-((1+sec(2*sigma))*(I_11-I01)*c(0, 0)**4)+2*sec(2*sigma)*(2*(np.cos(sigma))**2*I_10-I00)*c(0, 0)**2*c(1, 0)**2+(I_11-2*I00+sec(2*sigma)*(2*I_10+I_11-2*I00-I01)-I01)*c(1, 0)**4)
          * c(1, 1)+2*(np.cos(sigma))**2*sec(2*sigma)*(2*(np.cos(sigma))**2*I_11-I01)*c(0, 0)*(c(0, 0)**2-c(1, 0)**2)*c(1, 1)**2+2*sec(2*sigma)*((-2*(np.cos(sigma))**2*I_10+I00)*c(0, 0)**2-(I_10-2*(np.cos(sigma))**2*I00)*c(1, 0)**2)*c(1, 1)**3+2*(1+sec(2*sigma))*(I_10-I00)*c(0, 0)*c(1, 1)**4)))/((2*c(0, 0)*(c(0, 0)**2-c(1, 0)**2)*(c(1, 0)**2-c(1, 1)**2)))

    Ap = 2*u1/c(1, 0)**2
    Ar = 2*u2/c(0, 0)**2

    phi = np.arctan2(u3, u0)
    phi_ = np.arctan2(u1-np.sqrt((u0+u1+u2)**2-(u0+u2)**2-u3**2), u3)

    return Ap, Ar, phi, phi_, u0, u1, u2, u3


def P_11(I00, I_10, I01, I_11, sigma):

    # -1,1
    u0 = 1/(2*(c(0, 0)**2-c(1, 0)**2)*(c(1, 0)**2-c(1, 1)**2))*((c(0, 0)**2-c(1, 0)**2)*(I01*c(1, 0)**2-I_11*c(1, 1)**2) +
                                                                sec(2*sigma)*(c(1, 0)**2+c(1, 1)**2)*((-I_11+I01)*c(0, 0)**2+(2*I_10+I_11-2*I00-I01)*c(1, 0)**2+2*(-I_10+I00)*c(1, 1)**2))

    u1 = ((c(1, 0)**2*(-((1+sec(2*sigma))*(I_11-I01)*c(0, 0)**2)+sec(2*sigma)*((2*I_10+I_11+np.cos(2*sigma)*I_11-2*I00-2 *
          (np.cos(sigma))**2*I01)*c(1, 0)**2+2*(-I_10+I00)*c(1, 1)**2))))/((2*(-c(0, 0)**2+c(1, 0)**2)*(c(1, 0)**2-c(1, 1)**2)))

    u2 = (sec(2*sigma)*c(1, 1)**2*((I_11-I01)*c(0, 0)**2+(-2*I_10-I_11+2*I00+I01)*c(1, 0)
          ** 2+2*(I_10-I00)*c(1, 1)**2))/(2*(c(0, 0)**2-c(1, 0)**2)*(c(1, 0)**2-c(1, 1)**2))

    u3 = ((csc(2*sigma)*(2*(np.cos(sigma))**2*sec(2*sigma)*c(0, 0)*c(1, 0)**2*((I_11-2*(np.cos(sigma))**2*I01)*c(0, 0)**2-(2*I_10+I_11-2*I00-2*(np.cos(sigma))**2*I01)*c(1, 0)**2)+(-((1+sec(2*sigma))*(I_11-I01)*c(0, 0)**4)+2*sec(2*sigma)*(2*(np.cos(sigma))**2*I_10-I00)*c(0, 0)**2*c(1, 0)**2+(I_11-2*I00+sec(2*sigma)*(2*I_10+I_11-2*I00-I01)-I01)*c(1, 0)**4)
          * c(1, 1)+2*(np.cos(sigma))**2*sec(2*sigma)*(2*(np.cos(sigma))**2*I_11-I01)*c(0, 0)*(c(0, 0)**2-c(1, 0)**2)*c(1, 1)**2+2*sec(2*sigma)*((-2*(np.cos(sigma))**2*I_10+I00)*c(0, 0)**2-(I_10-2*(np.cos(sigma))**2*I00)*c(1, 0)**2)*c(1, 1)**3+2*(1+sec(2*sigma))*(I_10-I00)*c(0, 0)*c(1, 1)**4)))/((2*c(0, 0)*(c(0, 0)**2-c(1, 0)**2)*(c(1, 0)**2-c(1, 1)**2)))

    Ap = 2*u1/c(1, 0)**2
    Ar = 2*u2/c(0, 0)**2
    phi = np.arctan2(u3, u0)
    phi_ = np.arctan2(u1-np.sqrt((u0+u1+u2)**2-(u0+u2)**2-u3**2), u3)

    return Ap, Ar, phi, phi_, u0, u1, u2, u3


def P00_lat_right(I00, I_10, I01, I_11, a, b, c, sigma):

    u0 = (((1+a*(-1+b)*np.sqrt(b)-a*b*c)*I_10-a*I_11+(-np.sqrt(b)+c)*I_11+(-1+a*np.sqrt(b))*np.cos(2*sigma)*(-I_10+a*I_11)+2*np.sqrt(b)*I00-2*I01 +
          a**2*np.sqrt(b)*(I_11-2*b*I00+2*np.sqrt(b)*I01)))/((2*(-1+a*np.sqrt(b))*(-1+a*np.sqrt(b)+b-np.sqrt(b)*c+(-1+a*np.sqrt(b))*np.cos(2*sigma))))

    u1 = ((np.cos(2*sigma)*(np.sqrt(b)*I_10-I_11)-I_11+2*b*I00+np.sqrt(b)*(I_10-2*I01))
          )/((2*(-1+a*np.sqrt(b)+b-np.sqrt(b)*c+(-1+a*np.sqrt(b))*np.cos(2*sigma))))

    u2 = ((a*(-c*I_11-b*(I_10-2*a*I00)+2*I01+np.sqrt(b)*(c*I_10+I_11-2*(I00+a*I01))))) / \
        ((2*(-1+a*np.sqrt(b))*(-1+a*np.sqrt(b)+b -
         np.sqrt(b)*c+(-1+a*np.sqrt(b))*np.cos(2*sigma))))

    u3 = ((2*cot(sigma)*((-2+np.sqrt(b)*(1-a*(-2+b+c-np.sqrt(b)*c)))*I_10+(-1+np.sqrt(b)-c+a*(2-2*a*np.sqrt(b)+c))*I_11+(-1+a*np.sqrt(b))*(-2+2*a*b-np.sqrt(b)*(-2+c))*I00)-2*(-1+a*np.sqrt(b))*(np.sin(2*sigma)*(I_10-a*I_11) +
          ((-1+a)*(-1+np.sqrt(b))+(1+a*(-1+np.sqrt(b)))*np.cos(2*sigma))*csc(sigma)*sec(sigma)*I01+np.sqrt(b)*c*I00*np.tan(sigma))))/((4*(-1+a*np.sqrt(b))*(-1+a*np.sqrt(b)+b-np.sqrt(b)*c+(-1+a*np.sqrt(b))*np.cos(2*sigma))))

    Ap = 2*u1
    Ar = 2*u2
    phi = np.arctan2(u3, u0)
    phi_ = np.arctan2(u1-np.sqrt((u0+u1+u2)**2-(u0+u2)**2-u3**2), u3)

    return Ap, Ar, phi, phi_, u0, u1, u2, u3


def P00_cocientes(I00, I_10, I01, I_11, a, b, c, sigma):

    u0 = ((2-2*a*b*c)*I_10-(-1+a)*(1+c)*I_11-2*I00+2*a*b*c*I00-(-1+a)*np.cos(2*sigma) *
          (c*I_11-I01)-I01+a*(1+(-1+a)*b)*I01)/(2*(-1+a)*np.sqrt(b)*(a*b-c-(-1+c)*np.cos(2*sigma)))

    u1 = (2*(-1+a*b)*I_10+(-1+a)*I_11+2*I00+(-1+a)*np.cos(2*sigma) *
          (I_11-I01)+I01-a*(2*b*I00+I01))/(2*(-1+a)*(a*b-c-(-1+c)*np.cos(2*sigma)))

    u2 = (a*(2*(-1+c)*I_10+(-1+a)*I_11-2*(-1+c)*I00-(-1+a)*I01)) / \
        (2*(-1+a)*(a*b-c-(-1+c)*np.cos(2*sigma)))

    u3 = 1/(2*(-1+a)*np.sqrt(b)*(a*b-c-(-1+c)*np.cos(2*sigma)))*(-((-1+a)*c*np.sin(2*sigma)*I_11)+cot(sigma)*((-2+np.sqrt(b)*(1-a*(-2+b+2*c-2*np.sqrt(b)*c)))*I_10-(-1+a)*(-1+(1+a)*np.sqrt(b)-2*c)
                                                                                                              * I_11+(2+a**2*b**(3/2)-np.sqrt(b)*(2+a-2*c)-2*a*b*c)*I00-(-1+a)*(-1+np.sqrt(b))*(-1+a*np.sqrt(b))*I01-(-1+a)*np.cos(2*sigma)*I01)+np.sqrt(b)*(-1+a*b)*(-I_10+a*I00)*np.tan(sigma))

    # u0=prom_filter(u0)
    # u1=prom_filter(u1)
    # u2=prom_filter(u2)
    # u3=prom_filter(u3)

    Ap = 2*u1
    Ar = 2*u2
    phi = np.arctan2(u3, u0)
    phi_ = np.arctan2(u1-np.sqrt((u0+u1+u2)**2-(u0+u2)**2-u3**2), u3)

    return Ap, Ar, phi, phi_, u0, u1, u2, u3


def P_11_cocientes(I00, I_10, I01, I_11, a, b, c, sigma):

    u0 = (2*b*(-1+c**2)*I_10+(b*(-1+a-c)+c**2)*I_11-2*b*(-1+c**2)*I00-(b-c)*np.cos(2*sigma)
          * (c*I_11-I01)+(-c+b*(1+c-a*c))*I01)/(2*(-1+c)*(-a*b+c+(-b+c)*np.cos(2*sigma)))

    u1 = (-2*b*(-1+c)*I_10+(b-c)*I_11+2*b*(-1+c)*I00+(b-c)*np.cos(2*sigma)
          * (I_11-I01)+(-b+c)*I01)/(2*(-1+c)*(-a*b+c+(-b+c)*np.cos(2*sigma)))

    u2 = (b*c*(-2*(-1+c)*I_10-(-1+a)*I_11+2*(-1+c)*I00+(-1+a)*I01)) / \
        (2*(-1+c)*(-a*b+c+(-b+c)*np.cos(2*sigma)))

    u3 = 1/(2*(-1+c)*(-a*b+c+(-b+c)*np.cos(2*sigma)))*(c*(-b+c)*np.sin(2*sigma)*I_11+cot(sigma)*(np.sqrt(b)*c*(2*(-1+c)*I_10+a*(I_11-I01))+(-b+c)*np.cos(2*sigma)*I01+c*(-2*c *
                                                                                                                                                                         I_11+I01)+b**(3/2)*((-1+c)*I_10-I_11-(2+a)*(-1+c)*I00+I01)+b*(-2*(-1+c**2)*I_10+(1-a+2*c)*I_11+2*(-1+c**2)*I00+(-1+(-1+a)*c)*I01))+b**(3/2)*(-1+c)*(I_10-a*I00)*np.tan(sigma))

    #u0=prom_filter(u0)
    #u1=prom_filter(u1)
    #u2=prom_filter(u2)
    #u3=prom_filter(u3)

    Ap = 2*u1
    Ar = 2*u2
    #Ar = (2*(u0+u1+u2)-2*np.sqrt((u0+u1+u2) ** 2-(u0+u2)**2-u3**2))
    phi = np.arctan2(u3, u0)
    phi_ = np.arctan2(u1-np.sqrt((u0+u1+u2)**2-(u0+u2)**2-u3**2), u3)

    return Ap, Ar, phi, phi_, u0, u1, u2, u3


def Parameters_1(u0, u1, u2, u3):
    Ap = 2*u1/c(1, 0)**2
    Ar = (2*(u0+u1+u2)-2*np.sqrt((u0+u1+u2) ** 2-(u0+u2)**2-u3**2))/(c(0, 0)**2)
    phi = np.arctan2(u1-np.sqrt((u0+u1+u2)**2-(u0+u2)**2-u3**2), u3)
    return Ap, Ar, phi


def Parameters_2(u0, u1, u2, u3):
    Ap = 2*u1/c(1, 0)**2
    Ar = 2*u2/c(0, 0)**2
    phi = np.arctan2(u3, u0)
    return Ap, Ar, phi


def showfull():
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()
