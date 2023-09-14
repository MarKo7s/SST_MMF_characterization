from turtle import left
from numpy import *

def complexToRGB(x ,SV = 1, gamma_corr = False, background = 'black', transparency = 'off'):
    """ 
        IN-->
        x : complex array 
        SV: saturation
        gamma_corr: No linear intensity colormap
        background: Black or White -- Switch minimum intensity to white or black
        transparency: on or off -- on return RGB and alpha
        
        OUT-->
        rgb or rgba: rgb color mapping of the complex input array
        
        This mapping is fully custom, only uses numpy library
        
    """
    
    if gamma_corr == True:
        def gamma_curve(value,I = 5):
            return(1- e**(  -(value*I)**2))
        
        V = abs(x)/(abs(x)).max()
        V = gamma_curve(V)
        V = V/V.max()
    else:          
        V = abs(x) / (abs(x)).max() 
            
    C = V * SV
    T = C
    #Get Hue (from 0 to 360)
    H = angle(x,True) + 180
    Hp = H/60 #subdivide 6 different cases dependieng of the color
    X = C * (1 - abs(Hp%2 - 1))

    R1 = zeros(x.shape)
    G1 = copy(R1)
    B1 = copy(G1)

    #Mapping - 6 cases
    R1[((Hp>=0) & (Hp<=1)) | ((Hp>5) & (Hp<=6))] = C[((Hp>=0) & (Hp<=1)) | ((Hp>5) & (Hp<=6))]
    R1[((Hp>=1) & (Hp<=2)) | ((Hp>4) & (Hp<=5))] = X[((Hp>=1) & (Hp<=2)) | ((Hp>4) & (Hp<=5))]

    G1[((Hp>=0) & (Hp<=1)) | ((Hp>3) & (Hp<=4))] = X[((Hp>=0) & (Hp<=1)) | ((Hp>3) & (Hp<=4))]
    G1[ (Hp>1) &  (Hp<=3)] = C[ (Hp>1) &  (Hp<=3)]

    B1[((Hp>2) & (Hp<=3)) | ((Hp>5) & (Hp<=6))] = X[((Hp>2) & (Hp<=3)) | ((Hp>5) & (Hp<=6))]
    B1[(Hp>=3) & (Hp<=6)] = C[(Hp>=3) & (Hp<=6)]

    #intensity correction to each component
    m = V-C
    
    R = R1+m
    G = G1+m
    B = B1+m
    
    #Invert Black to White (background)
    if background == 'white':
        R = (R - 1) * (-1)
        G = (G - 1) * (-1)
        B = (B - 1) * (-1)
    
    if transparency == 'on':
        rgb = stack((R ,G ,B, T), axis = 2) #faster
    else:
        rgb = stack((R ,G ,B), axis = 2) #faster

    return(rgb)


def RGB_colorbar(ax, A, P, fontsize = 20, label = 'I', labelpad = None, orientation = 'vertical'):
    
    if orientation != 'horizontal':
        if labelpad == None:
            labelpad = -70
        RGB_colorbar_vertical(ax = ax, A = A, P = P, label = label, fontsize = fontsize, labelpad  = labelpad)
    else:
        if labelpad == None:
            labelpad = 0
        RGB_colorbar_horizontal(ax = ax, A = A, P = P, label = label, fontsize = fontsize, labelpad  = labelpad)


def Intensity_colorbar(ax, H, W, cmap = 'Turbo',  fontsize = 20, label = 'I', labelpad = None, orientation = 'vertical'):
    
    if orientation != 'horizontal':
        if labelpad == None:
            labelpad = -70
        Intensity_colorbar_vertical(ax = ax, H = H, W = W, cmap = cmap, label = label, fontsize = fontsize, labelpad  = labelpad)
    else:
        if labelpad == None:
            labelpad = 0
        Intensity_colorbar_horizontal(ax = ax, H = H, W = W, cmap = cmap, label = label, fontsize = fontsize, labelpad  = labelpad)


def RGB_colorbar_vertical(ax, A, P, label = 'I', labelpad = -70, fontsize = 20 ):
    """ 
        ax = axes here you want to plot the RGB, trated as a regular figure
        A = size of the amplitude in sample - equivalent to height
        P = size of the phase in samples - equivalent to width
        labelpad = some manual padding of the labels
    """
    Asamp = A
    Phsamp = P
    ph = linspace(-pi,pi,Phsamp) #xaxis
    a = linspace(0.000001,1,Asamp) #yaxis
    a,b = meshgrid(a,a)
    A = flipud(b[:,0:Phsamp])
    custBar = A*exp(1j*(ph))

    ax.imshow(complexToRGB(custBar))
    ax.set_xticks(ax.get_xbound())
    ax.set_yticks(ax.get_ybound())
    ylabels = ['1', '0']
    xlabels = ['$-\pi$', '$\pi$']

    labelsX = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(len(labelsX)) :
        labelsX[i] =  xlabels[i] 

    labelsY = [item.get_text() for item in ax.get_yticklabels()]
    for i in range(len(ylabels)) :
        labelsY[i] = ylabels[i]

    ax.set_xticklabels(labelsX,fontsize = fontsize)
    ax.set_yticklabels(labelsY,fontsize = fontsize)

    ax.tick_params(axis = 'y', direction='in', left = 'off', right = 'on',
                   labelright = True,labelleft=False)
    
    ax.tick_params(bottom = False)

    ax.set_xlabel('$\phi$',fontsize = fontsize)
    ax.set_ylabel(f'${label}$',fontsize = fontsize,labelpad = labelpad)
    

def RGB_colorbar_horizontal(ax, A, P, label = 'I', labelpad = 0, fontsize = 20 ):
    """ 
        ax = axes here you want to plot the RGB, trated as a regular figure
        A = size of the amplitude in sample - equivalent to height
        P = size of the phase in samples - equivalent to width
        labelpad = some manual padding of the labels
    """
    Asamp = A
    Phsamp = P
    ph = linspace(-pi,pi,Phsamp) #xaxis
    a = linspace(0.000001,1,Asamp) #yaxis
    a,b = meshgrid(a,a)
    A = transpose((b[:,0:Phsamp]))
    custBar = A*exp(1j*(ph[:,None]))

    ax.imshow(complexToRGB(custBar, gamma_corr = True))
    ax.set_xticks(ax.get_xbound())
    ax.set_yticks(ax.get_ybound())
    xlabels = ['0', '1']
    ylabels = ['$\pi$', '$-\pi$']

    labelsX = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(len(labelsX)) :
        labelsX[i] =  xlabels[i] 

    labelsY = [item.get_text() for item in ax.get_yticklabels()]
    for i in range(len(ylabels)) :
        labelsY[i] = ylabels[i]

    ax.set_xticklabels(labelsX,fontsize = fontsize)
    ax.set_yticklabels(labelsY,fontsize = fontsize)

    ax.tick_params(axis = 'y', direction='in',left = 'on', right = 'off',
                   labelright = False,labelleft=True, labelrotation = 0)

    ax.tick_params(bottom = False)
    
    ax.set_ylabel('$\phi$',fontsize = fontsize, rotation = 'horizontal', rotation_mode = 'anchor', ha = 'left', va = 'center_baseline')
    ax.set_xlabel(f'${label}$',fontsize = fontsize, labelpad = labelpad)


def Intensity_colorbar_vertical(ax, H,W, cmap = 'turbo', label = 'I', labelpad = -70, fontsize = 20 ):
    """ 
        ax = axes here you want to plot the RGB, trated as a regular figure
        A = size of the Intensity in pixels - equivalent to height
        cmap = matplotlib calor mapping
        labelpad = some manual padding of the labels
    """
    w = linspace(0,1, W) #xaxis
    h = linspace(0.000001,1, H) #yaxis
    a,b = meshgrid(w,h)
    custBar = flipud(b)

    ax.imshow(custBar, cmap = cmap)
    ax.set_xticks(ax.get_xbound())
    ax.set_yticks(ax.get_ybound())
    ylabels = ['1', '0']

    ax.set_xticklabels(())

    labelsY = [item.get_text() for item in ax.get_yticklabels()]
    for i in range(len(ylabels)) :
        labelsY[i] = ylabels[i]

    ax.set_yticklabels(labelsY,fontsize = fontsize)

    ax.tick_params(axis = 'y', direction='in',left = 'off', right = 'on',
                   labelright = True, labelleft=False)
    
    ax.tick_params(bottom = False)
    
    ax.set_ylabel(f'${label}$',fontsize = fontsize,labelpad = labelpad)
    
def simple_polar_RGB_colorbar(ax, N, gamma_corr = False, background = 'black', transparency = 'off'):
    
    def cart2pol(x, y):
        rho = sqrt(x**2 + y**2)
        phi = arctan2(y, x)
        return(rho, phi)
    
    x = linspace(-1, 1, N)
    y = linspace(-1,1, N)
    X,Y = meshgrid(x,y)
    rho,phi = cart2pol(X,Y)
    custBar = rho*exp(1j*(phi))
    custBar[rho>1] = 0
    rgb = complexToRGB(custBar, gamma_corr = gamma_corr, background = background, transparency = transparency )
    ax.imshow(rgb)
    
    ax.set_axis_off()
    
def Intensity_colorbar_horizontal(ax, H, W, cmap = 'turbo', label = 'I', labelpad = 0, fontsize = 20 ):
    """ 
        ax = axes here you want to plot the RGB, trated as a regular figure
        A = size of the Intensity in pixels - equivalent to height
        cmap = matplotlib calor mapping
        labelpad = some manual padding of the labels
    """
    w = linspace(0,1, W) #xaxis
    h = linspace(0.000001,1, H) #yaxis
    a,b = meshgrid(w,h)
    custBar = transpose(b)

    ax.imshow(custBar, cmap = cmap)
    ax.set_xticks(ax.get_xbound())
    ax.set_yticks(ax.get_ybound())
    xlabels = ['0', '1']

    ax.set_yticklabels(())

    labelsX = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(len(xlabels)) :
        labelsX[i] = xlabels[i]

    ax.set_xticklabels(labelsX, fontsize = fontsize)

    ax.tick_params(axis = 'y', direction='in',left = 'off', right = 'on',
                   labelright = True, labelleft=False)
    
    ax.tick_params(bottom = False)
    
    ax.set_xlabel(f'${label}$',fontsize = fontsize,labelpad = labelpad)
       