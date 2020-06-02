# -*- coding: utf-8 -*-

######################################################################
######################################################################
######################################################################
#                                                                    #
# degrade.py                                                         #
#                                                                    #
# Tanaus\'u del Pino Alem\'an                                        #
#   Instituto de Astrof\'isica de Canarias                           #
#                                                                    #
######################################################################
######################################################################
#                                                                    #
# Class for image degradation                                        #
#                                                                    #
######################################################################
######################################################################
#                                                                    #
######################################################################
######################################################################
#                                                                    #
#  02/06/2020 - V0.0.1 - Added some more unit changes. (TdPA)        #
#                      - Bugfix: The PSF for gauss and square for    #
#                        the 1D filter was using the wrong center    #
#                        variable, the input instead of the local    #
#                        one. (TdPA)                                 #
#                                                                    #
#  26/04/2018 - V0.0.0 - Start code. (TdPA)                          #
#                                                                    #
######################################################################
######################################################################
######################################################################

import sys,copy,struct
try:
    import numpy as np
except ImportError:
    print('##Error## Missing numpy')
    raise
except:
    raise
try:
    from scipy import fftpack
    from scipy import signal
    from scipy import interpolate
    from scipy.ndimage import gaussian_filter
except ImportError:
    print('##Error## Missing scipy')
    raise
except:
    raise

######################################################################
######################################################################
######################################################################
######################################################################

class degrade_class():
    ''' Class with functions to degrade images
    '''

    def __init__(self):
        ''' Initialize class
        '''

        # Useful constants
        self.__cm2s = 1.3787950676165365e-8 
        self.__fwhm2sig = 4.24660900144009534e-1
        self.__sqrt2pi = 2.50662827463100024e0
        self.__deg2ra = 1.74532925199432955e-2
        self.__ra2deg = 5.72957795130823229e1

######################################################################
######################################################################

    def __error(self, msg=None):
        ''' Error exit
        '''

        if msg is None: msg = ''

        print('##ERROR## '+msg)

        return -1

######################################################################
######################################################################

    def cm2sec(self, x):

        return x*self.__cm2s

######################################################################
######################################################################

    def m2sec(self, x):

        return x*self.__cm2s*1e2

######################################################################
######################################################################

    def km2sec(self, x):

        return x*self.__cm2s*1e5

######################################################################
######################################################################

    def Mm2sec(self, x):

        return x*self.__cm2s*1e8

######################################################################
######################################################################

    def sec2cm(self, x):

        return x/self.__cm2s

######################################################################
######################################################################

    def sec2Mm(self, x):

        return x*1e-8/self.__cm2s
######################################################################
######################################################################

    def sec2Mm(self, x):

        return x*1e-8/self.__cm2s

######################################################################
######################################################################

    def sec2Mm(self, x):

        return x*1e-8/self.__cm2s

######################################################################
######################################################################

    def fwhm2sig(self, x):

        return x*self.__fwhm2sig

######################################################################
######################################################################

    def deg2ra(self, x):

        return x*self.__deg2ra

######################################################################
######################################################################

    def sec2ra(self, x):

        return x*self.__deg2ra/3600.

######################################################################
######################################################################

    def ra2deg(self, x):

        return x*self.__ra2deg

######################################################################
######################################################################

    def ra2sec(self, x):

        return x*self.__ra2deg*3600.

######################################################################
######################################################################

    def ra2deg(self, x):

        return x*self.__ra2deg

######################################################################
######################################################################

    def nm2m(self, x):

        return x*1e-9

######################################################################
######################################################################

    def Ang2m(self, x):

        return x*1e-10

######################################################################
######################################################################

    def m2Ang(self, x):

        return x*1e10

######################################################################
######################################################################

    def mAng2m(self, x):

        return x*1e-13

######################################################################
######################################################################

    def convol2D(self, data, xaxis=None, yaxis=None, xpar=None, \
                 ypar=None, xperiod=None, yperiod=None, \
                 xextend=None, yextend=None, xdir=None, ydir=None, \
                 kernel=None, r0=None, D=None, l0=None, \
                 xequi=None, yequi=None):
        ''' Applies a 2D convolution
        '''

        #
        # Manage arguments
        #

        # Data
        try:

            idata = copy.deepcopy(data)

            if len(data.shape) != 2:
                self.__error('data must be 2D array')

            if 'float' not in str(data.dtype):
                self.__error('data must be float type')

        except AttributeError:

            try:

                idata = copy.deepcopy(data)

                idata = np.array(idata)

                if len(idata.shape) != 2:
                    self.__error('data must be 2D array')
                    return -1

                if 'float' not in str(idata.dtype):
                    self.__error('data must be float type')
                    return -1

            except:

                self.__error('unexpected error while making ' + \
                             'an array from the data')
                return -1

        except:

            self.__error('unexpected error while checking ' + \
                         'the data')
            return -1

        NX, NY = idata.shape

        if NX < 2:
            self.__error('X size must be larger than 1')
            return -1
        if NY < 2:
            self.__error('Y size must be larger than 1')
            return -1

        # xaxis
        if xaxis is None:

            ixaxis = np.linspace(0.,1.*(NX-1),num=NX)

        else:

            try:

                ixaxis = copy.deepcopy(xaxis)

                if len(ixaxis.shape) != 1:
                    self.__error('xaxis must be 1D array')
                    return -1

                if 'float' not in str(ixaxis.dtype):
                    self.__error('xaxis must be float type')
                    return -1

                if ixaxis.shape[0] != NX:
                    self.__error('dimension mismatch between ' + \
                                 'data and xaxis')
                    return -1

            except AttributeError:

                try:

                    ixaxis = copy.deepcopy(xaxis)

                    ixaxis = np.array(ixaxis)

                    if len(ixaxis.shape) != 1:
                        self.__error('xaxis must be 1D array')
                        return -1

                    if 'float' not in str(ixaxis.dtype):
                        self.__error('xaxis must be float type')
                        return -1

                    if ixaxis.shape[0] != NX:
                        self.__error('dimension mismatch between ' + \
                                     'data and xaxis')
                        return -1

                except:

                    self.__error('unexpected error while making ' + \
                                 'an array from xaxis')
                    return -1

            except:

                self.__error('unexpected error while checking ' + \
                             'xaxis')
                return -1


        # yaxis
        if yaxis is None:

            iyaxis = np.linspace(0.,1.*(NY-1),num=NY)

        else:

            try:

                iyaxis = copy.deepcopy(yaxis)

                if len(iyaxis.shape) != 1:
                    self.__error('yaxis must be 1D array')
                    return -1

                if 'float' not in str(iyaxis.dtype):
                    self.__error('yaxis must be float type')
                    return -1

                if iyaxis.shape[0] != NY:
                    self.__error('dimension mismatch between ' + \
                                 'data and yaxis')
                    return -1

            except AttributeError:

                try:

                    iyaxis = copy.deepcopy(yaxis)

                    iyaxis = np.array(iyaxis)

                    if len(iyaxis.shape) != 1:
                        self.__error('yaxis must be 1D array')
                        return -1

                    if 'float' not in str(iyaxis.dtype):
                        self.__error('yaxis must be float type')
                        return -1

                    if iyaxis.shape[0] != NY:
                        self.__error('dimension mismatch between ' + \
                                     'data and yaxis')
                        return -1

                except:

                    self.__error('unexpected error while making ' + \
                                 'an array from yaxis')
                    return -1

            except:

                self.__error('unexpected error while checking ' + \
                             'yaxis')
                return -1


        # xperiod
        if xperiod is None:

            ixperiod = False

        else:

            ixperiod = xperiod

            if not isinstance(ixperiod, bool):
                self.__error('xperiod must be bool')
                return -1


        # yperiod
        if yperiod is None:

            iyperiod = False

        else:

            iyperiod = yperiod

            if not isinstance(iyperiod, bool):
                self.__error('yperiod must be bool')
                return -1


        # xextend
        if xextend is None:

            ixextend = False

        else:

            ixextend = xextend

            if not isinstance(ixextend, bool):
                self.__error('xextend must be bool')
                return -1

            if ixperiod and ixextend:
                self.__error('xextend and xperiod cannot ' + \
                             'be True at the same time')
                return -1


        # yextend
        if yextend is None:

            iyextend = False

        else:

            iyextend = yextend

            if not isinstance(iyextend, bool):
                self.__error('yextend must be bool')
                return -1

            if iyperiod and iyextend:
                self.__error('yextend and yperiod cannot ' + \
                             'be True at the same time')
                return -1


        # xequi
        if xequi is None:

            ixequi = False

        else:

            ixequi = xequi

            if not isinstance(ixequi, bool):
                self.__error('xequi must be bool')
                return -1


        # yequi
        if yequi is None:

            iyequi = False

        else:

            iyequi = yequi

            if not isinstance(iyequi, bool):
                self.__error('yequi must be bool')
                return -1


        # xdir
        if xdir is None:

            ixdir = 5

        else:

            ixdir = xdir

            if not isinstance(ixdir, float):
                self.__error('xdir must be a float number')
                return -1


        # ydir
        if ydir is None:

            iydir = 5

        else:

            iydir = ydir

            if not isinstance(iydir, float):
                self.__error('ydir must be a float number')
                return -1


        # Kernel
        vkernel = ['gauss','square','airy','seeing']
        if kernel is None:

            tkernel = 0

        else:

            tkernel = -1

            for ii,key in zip(range(len(vkernel)),vkernel):

                if kernel.lower() in key:
                    tkernel = ii
                    break

            if tkernel < 0:
                self.__error('Type of kernel not recognized')
                return -1

            if tkernel == 0 or tkernel == 1:

                # xpar
                if xpar is None:

                    self.__error('xpar is required for kernel ' + \
                                 vkernel[tkernel])
                    return -1

                else:

                    ixpar = xpar

                    if not isinstance(ixpar, float):
                        self.__error('xpar must be a float number')
                        return -1


                # ypar
                if ypar is None:

                    self.__error('ypar is required for kernel ' + \
                                 vkernel[tkernel])
                    return -1

                else:

                    iypar = ypar

                    if not isinstance(iypar, float):
                        self.__error('ypar must be a float number')
                        return -1

            if tkernel == 2 or tkernel == 3:

                if l0 is None:

                    self.__error('l0 is required for kernel ' + \
                                 vkernel[tkernel])
                    return -1

                else:

                    il0 = l0

                    if not isinstance(il0, float):
                        self.__error('l0 must be a float number')
                        return -1

                if D is None:

                    self.__error('D is required for kernel ' + \
                                 vkernel[tkernel])
                    return -1

                else:

                    iD = D

                    if not isinstance(iD, float):
                        self.__error('D must be a float number')
                        return -1

                if tkernel == 3:

                    if r0 is None:

                        self.__error('r0 is required for kernel ' + \
                                     vkernel[tkernel])
                        return -1

                    else:

                        ir0 = r0

                        if not isinstance(ir0, float):
                            self.__error('r0 must be a float number')
                            return -1


        #
        # Size check
        #
        if tkernel == 0 or tkernel == 2 or tkernel == 3:

            if xpar is not None:
                xdis = ixdir*xpar
            else:
                xdis = ixdir

            if ypar is not None:
                ydis = iydir*ypar
            else:
                ydis = iydir

        elif tkernel == 1:

            xdis = xpar*.5
            ydis = ypar*.5


        # Shift axis to 0 origin
        ixaxis -= ixaxis[0]
        iyaxis -= iyaxis[0]


        # Check if equidistant grid
        dx = ixaxis[1:] - ixaxis[0:-1]

        if np.min(dx) <= 0.:
            self.__error('xaxis must have only unique values')
            return -1

        if not ixequi:
            ixequi = True
            for ii in range(0,len(dx)-1):
                for jj in range(ii+1,len(dx)):
                    if (dx[ii] - dx[jj])/(dx[ii] + dx[jj]) > 1e-8:
                        ixequi = False
                        break
                if not ixequi:
                    break

        dy = iyaxis[1:] - iyaxis[0:-1]

        if np.min(dy) <= 0.:
            self.__error('yaxis must have only unique values')
            return -1

        if not iyequi:
            iyequi = True
            for ii in range(0,len(dy)-1):
                for jj in range(ii+1,len(dy)):
                    if (dy[ii] - dy[jj])/(dy[ii] + dy[jj]) > 1e-8:
                        iyequi = False
                        break
                if not iyequi:
                    break

        #
        # Manage axis
        #

        # X
        if ixequi:
            NXi = NX
            dsx = dx[0]
            xaxisi = np.linspace(0,NXi-1,num=NXi,dtype=ixaxis.dtype)
        else:
            dsx = np.amin(dx)
            NXi = int((xaxis[-1] - xaxis[0])/dsx + 1)
            xaxisi = np.linspace(0,NXi-1,num=NXi,dtype=ixaxis.dtype)

        if iyequi:
            NYi = NY
            dsy = dy[0]
            yaxisi = np.linspace(0,NYi-1,num=NYi,dtype=iyaxis.dtype)
        else:
            dsy = np.amin(dy)
            NYi = int((yaxis[-1] - yaxis[0])/dsy + 1)
            yaxisi = np.linspace(0,NYi-1,num=NYi,dtype=iyaxis.dtype)

        # Domain sizes
        DX = xaxisi[-1] - xaxisi[0]
        DY = yaxisi[-1] - yaxisi[0]
        DX *= dsx
        DY *= dsy

        # Interpolate
        if not ixequi or not iyequi:

            if (not ixequi) and (not iyequi):

                xx, yy = np.meshgrid(xaxisi*dsx, yaxisi*dsy, \
                                     indexing='ij')

            elif not ixequi:

                xx, yy = np.meshgrid(xaxisi*dsx, iyaxis, \
                                     indexing='ij')

            elif not iyequi:

                xx, yy = np.meshgrid(ixaxis, yaxisi*dsy, \
                                     indexing='ij')

            r = interpolate.RegularGridInterpolator((ixaxis,iyaxis), \
                                                    idata)

            # Interpolate into regular grid
            idata = r((xx,yy))


        #
        # Check padding
        #

        NXe = 0
        NYe = 0

        # If extending the borders as constants
        if ixextend or iyextend:

            if ixextend and iyextend:

                NXe = int(xdis/dsx + 1)
                NYe = int(ydis/dsy + 1)

            elif xextend:

                NXe = int(xdis/dsx + 1)

            elif yextend:

                NYe = int(ydis/dsy + 1)

            idata = np.pad(idata,((NXe,NXe),(NYe,NYe)),'edge')

        if ixperiod or iyperiod:

            if ixperiod and iyperiod:

                if xdis > DX:
                    NXe = int((xdis - DX)/dsx + 1)
                if ydis > DY:
                    NYe = int((ydis - DY)/dsy + 1)

            elif ixperiod:

                if xdis > DX:
                    NXe = int((xdis - DX)/dsx + 1)

            elif iyperiod:

                if ydis > DY:
                    NYe = int((ydis - DY)/dsy + 1)

            if NXe > 0 or NYe > 0:
                idata = np.pad(idata,((NXe,NXe),(NYe,NYe)),'wrap')

        idata = np.fft.fft2(idata)

        NXc, NYc = idata.shape

        # Build frequency axis
        u = np.fft.fftfreq(NXc, d=dsx)
        v = np.fft.fftfreq(NYc, d=dsy)

        uu, vv = np.meshgrid(u,v,indexing='ij')

        # Build kernel

        # MTF gauss
        if tkernel == 0:

            MTF = np.exp(-2.*uu*uu*np.pi*np.pi*ixpar*ixpar)* \
                  np.exp(-2.*vv*vv*np.pi*np.pi*iypar*iypar)/dsx/dsy

        # MTF square
        elif tkernel == 1:

            MTF = np.sinc(ixpar*uu)*np.sinc(iypar*vv)/ \
                  dsx/dsy

        # MTF Airy
        elif tkernel == 2:

            uv = self.ra2sec(l0*np.sqrt(uu*uu + vv*vv)/D)

            uv = np.clip(uv, a_min=-1, a_max=1.)

            MTF = 2.*(np.arccos(uv) - uv*np.sqrt(1.-uv*uv))/ \
                  (np.pi*dsx*dsy)

        # MTF Fried (1966), Long-Exposure approximation
        elif tkernel == 3:

            uv = self.ra2sec(l0*np.sqrt(uu*uu + vv*vv)/D)

            uv = np.clip(uv, a_min=-1., a_max=1.)

            MTF = np.exp(-3.44*np.power(uv*D/r0,5./3.))* \
                  2.*(np.arccos(uv) - uv*np.sqrt(1.-uv*uv))/ \
                  (np.pi*dsx*dsy)

        idata *= MTF

        idata = np.fft.ifft2(idata).real*dsx*dsy

        if NXe > 0:
            idata = idata[:NXe+NXi,:]
            idata = idata[NXe:,:]
        if NYe > 0:
            idata = idata[:,:NYe+NYi]
            idata = idata[:,NYe:]

        # Return to original dimensions
        if ixequi or iyequi:

            return idata

        else:

            xx, yy = np.meshgrid(ixaxis, iyaxis, indexing='ij')

            if (not ixequi) and (not iyequi):

                dims = (xaxisi*dsx, yaxisi*dsy)

            elif not ixequi:

                dims = (xaxisi*dsx, iyaxis)

            elif not iyequi:

                dims = (ixaxis, yaxisi*dsy)

            r = interpolate.RegularGridInterpolator(dims, idata)

            # Interpolate to original grid
            return r((xx,yy))

######################################################################
######################################################################

    def convol1D(self, data, axis=None, par=None, period=None, \
                 extend=None, adir=None, kernel=None, equi=None, \
                 kernel_file=None):
        ''' Applies a 1D convolution
        '''

        #
        # Manage arguments
        #

        # Data
        try:

            idata = copy.deepcopy(data)

            if len(data.shape) != 1:
                self.__error('data must be 1D array')

            if 'float' not in str(data.dtype):
                self.__error('data must be float type')

        except AttributeError:

            try:

                idata = copy.deepcopy(data)

                idata = np.array(idata)

                if len(idata.shape) != 1:
                    self.__error('data must be 1D array')
                    return -1

                if 'float' not in str(idata.dtype):
                    self.__error('data must be float type')
                    return -1

            except:

                self.__error('unexpected error while making ' + \
                             'an array from the data')
                return -1

        except:

            self.__error('unexpected error while checking ' + \
                         'the data')
            return -1

        NN = idata.shape[0]

        if NN < 2:
            self.__error('data size must be larger than 1')
            return -1

        # axis
        if axis is None:

            iaxis = np.linspace(0.,1.*(NN-1),num=NN)

        else:

            try:

                iaxis = copy.deepcopy(axis)

                if len(iaxis.shape) != 1:
                    self.__error('axis must be 1D array')
                    return -1

                if 'float' not in str(iaxis.dtype):
                    self.__error('axis must be float type')
                    return -1

                if iaxis.shape[0] != NN:
                    self.__error('dimension mismatch between ' + \
                                 'data and axis')
                    return -1

            except AttributeError:

                try:

                    iaxis = copy.deepcopy(axis)

                    iaxis = np.array(iaxis)

                    if len(iaxis.shape) != 1:
                        self.__error('axis must be 1D array')
                        return -1

                    if 'float' not in str(iaxis.dtype):
                        self.__error('axis must be float type')
                        return -1

                    if iaxis.shape[0] != NN:
                        self.__error('dimension mismatch between ' + \
                                     'data and axis')
                        return -1

                except:

                    self.__error('unexpected error while making ' + \
                                 'an array from axis')
                    return -1

            except:

                self.__error('unexpected error while checking ' + \
                             'axis')
                return -1


        # period
        if period is None:

            iperiod = False

        else:

            iperiod = period

            if not isinstance(iperiod, bool):
                self.__error('period must be bool')
                return -1


        # extend
        if extend is None:

            iextend = False

        else:

            iextend = extend

            if not isinstance(iextend, bool):
                self.__error('extend must be bool')
                return -1

            if iperiod and iextend:
                self.__error('extend and period cannot ' + \
                             'be True at the same time')
                return -1


        # adir
        if adir is None:

            idir = 5

        else:

            idir = adir

            if not isinstance(idir, float):
                self.__error('adir must be a float number')
                return -1


        # equi
        if equi is None:

            iequi = False

        else:

            iequi = equi

            if not isinstance(iequi, bool):
                self.__error('equi must be a bool')
                return -1


        # Kernel
        vkernel = ['gauss','square','custom']
        if kernel is None:

            tkernel = 0

        else:

            tkernel = -1

            for ii,key in zip(range(len(vkernel)),vkernel):

                if kernel.lower() in key:
                    tkernel = ii
                    break

            if tkernel < 0:
                self.__error('Type of kernel not recognized')
                return -1

            if tkernel == 0 or tkernel == 1:

                # par
                if par is None:

                    self.__error('par is required for kernel ' + \
                                 vkernel[tkernel])
                    return -1

                else:

                    ipar = par

                    if not isinstance(ipar, float):
                        self.__error('par must be a float number')
                        return -1

            elif tkernel == 2:

                # kernel_file
                if kernel_file is None:

                    self.__error('kernel_file is required for ' + \
                                 'kernel ' + vkernel[tkernel])
                    return -1

                else:

                    ikernel_file = kernel_file

                    if not isinstance(ikernel_file, str):
                        self.__error('kernel_file must be a string')
                        return -1

                # Read Kernel
                try:
                    f_kernel = open(ikernel_file, 'rb')
                except IOError:
                    self.__error('kernel_file not found')
                    return -1
                except:
                    msg = 'problem opening kernel file\n' + \
                          sys.exc_info()[0] + '\n' + \
                          sys.exc_info()[1]
                    self.__error(msg)
                    return -1

                bytes = f_kernel.read(4)
                nl = int(struct.unpack('i', bytes)[0])
                bytes = f_kernel.read(8*nl)
                l_PSF = np.array(struct.unpack('d'*nl, bytes), \
                                 dtype=np.float64)
                bytes = f_kernel.read(8*nl)
                PSF = np.array(struct.unpack('d'*nl, bytes), \
                               dtype=np.float64)
                f_kernel.close()

                # Shift axis to 0 origin
                l_PSF -= l_PSF[0]

        #
        # Size check
        #
        if tkernel == 0:

            if par is not None:
                dis = idir*par
            else:
                dis = idir

        elif tkernel == 1:

            dis = par*.5

        elif tkernel == 2:

            dis = l_PSF[-1]

        # Shift axis to 0 origin
        iaxis -= iaxis[0]

        # Check if equidistant grid
        ds = iaxis[1:] - iaxis[0:-1]

        if np.min(ds) <= 0.:
            self.__error('axis must have only unique values')
            return -1

        if not iequi:
            iequi = True
            for ii in range(0,len(ds)-1):
                for jj in range(ii+1,len(ds)):
                    if (ds[ii] - ds[jj])/(ds[ii] + ds[jj]) > 1e-8:
                        iequi = False
                        break
                if not iequi:
                    break

        #
        # Manage axis
        #

        if iequi:
            NNi = NN
            dss = ds[0]
            axisi = np.linspace(0,NNi-1,num=NNi,dtype=iaxis.dtype)
        else:
            dss = np.amin(ds)
            NNi = int((axis[-1] - axis[0])/dss + 1)
            axisi = np.linspace(0,NNi-1,num=NNi,dtype=iaxis.dtype)

        # Domain sizes
        DD = axisi[-1] - axisi[0]
        DD *= dss


        # Interpolate
        if not iequi:

             idata = np.interp(axisi*dss, iaxis, idata)


        #
        # Check padding
        #

        NNe = 0

        # If extending the borders as constants
        if iextend:

            NNe = int(dis/dss + 1)
            idata = np.pad(idata,((NNe,NNe)),'edge')

        if iperiod:

           if dis > DD:
                NNe = int((dis - DD)/dss + 1)
                idata = np.pad(idata,((NNe,NNe)),'wrap')

        idata = np.fft.fft(idata)

        NNc = idata.shape[0]

        # Build frequency axis
        u = np.fft.fftfreq(NNc, d=dss)

        # Build kernel

        # MTF gauss
        if tkernel == 0:

            MTF = np.exp(-2.*u*u*np.pi*np.pi*ipar*ipar)/dss
            ifact = dss

        # MTF square
        elif tkernel == 1:

           #MTF = ipar*np.sinc(ipar*u)/dss
            MTF = np.sinc(ipar*u)/dss
            ifact = dss

        # Custom MTF
        elif tkernel == 2:

            # Integrate PSF to 1
            ipar = np.trapz(PSF,l_PSF)
            PSF *= dss/ipar

            # Get the sizes of axis and data
            Dl0 = axisi[-1]*dss
            Dl = (idata.size-1)*dss

            # Compute how much out for one side
            Dout = (l_PSF[-1] - Dl)*.5

            # Extended axis
            axisi_tmp = np.linspace(0, idata.size-1, \
                                    num=idata.size, \
                                    dtype=iaxis.dtype)

            # Shift axis
            axisi_tmp *= dss
            axisi_tmp += Dout

            # Inteprolate
            PSF = np.interp(axisi_tmp, l_PSF, PSF, \
                            left=0., right=0.)

            # Free
            axisi_tmp = []

            MTF = np.fft.fft(PSF)
            MTF *= np.conj(MTF)
            MTF = np.sqrt(MTF)

            ifact = 1e0

        idata *= MTF

        idata = np.fft.ifft(idata).real*ifact

        if NNe > 0:
            idata = idata[:NNe+NNi]
            idata = idata[NNe:]

        # Return to original dimensions
        if iequi:

            return idata

        else:

            # Interpolate to original grid
            return np.interp(iaxis, axisi*dss, idata)

######################################################################
######################################################################

    def filter1D(self, data, axis=None, par=None, center=None, \
                 extend=None, kernel=None, equi=None, \
                 kernel_file=None, norm=None):
        ''' Applies a 1D convolution
        '''

        #
        # Manage arguments
        #

        # Data
        try:

            idata = copy.deepcopy(data)

            if len(data.shape) != 1:
                self.__error('data must be 1D array')

            if 'float' not in str(data.dtype):
                self.__error('data must be float type')

        except AttributeError:

            try:

                idata = copy.deepcopy(data)

                idata = np.array(idata)

                if len(idata.shape) != 1:
                    self.__error('data must be 1D array')
                    return -1

                if 'float' not in str(idata.dtype):
                    self.__error('data must be float type')
                    return -1

            except:

                self.__error('unexpected error while making ' + \
                             'an array from the data')
                return -1

        except:

            self.__error('unexpected error while checking ' + \
                         'the data')
            return -1

        NN = idata.shape[0]

        if NN < 2:
            self.__error('data size must be larger than 1')
            return -1

        # axis
        if axis is None:

            self.__error('the abscissa is required')
            return -1

        else:

            try:

                iaxis = copy.deepcopy(axis)

                if len(iaxis.shape) != 1:
                    self.__error('axis must be 1D array')
                    return -1

                if 'float' not in str(iaxis.dtype):
                    self.__error('axis must be float type')
                    return -1

                if iaxis.shape[0] != NN:
                    self.__error('dimension mismatch between ' + \
                                 'data and axis')
                    return -1

            except AttributeError:

                try:

                    iaxis = copy.deepcopy(axis)

                    iaxis = np.array(iaxis)

                    if len(iaxis.shape) != 1:
                        self.__error('axis must be 1D array')
                        return -1

                    if 'float' not in str(iaxis.dtype):
                        self.__error('axis must be float type')
                        return -1

                    if iaxis.shape[0] != NN:
                        self.__error('dimension mismatch between ' + \
                                     'data and axis')
                        return -1

                except:

                    self.__error('unexpected error while making ' + \
                                 'an array from axis')
                    return -1

            except:

                self.__error('unexpected error while checking ' + \
                             'axis')
                return -1


        # extend
        if extend is None:

            iextend = False

        else:

            iextend = extend

            if not isinstance(iextend, bool):
                self.__error('extend must be bool')
                return -1


        # equi
        if equi is None:

            iequi = False

        else:

            iequi = equi

            if not isinstance(iequi, bool):
                self.__error('equi must be a bool')
                return -1


        # norm
        if norm is None:

            inorm = False

        else:

            inorm = norm

            if not isinstance(inorm, bool):
                self.__error('norm must be a bool')
                return -1


        # Kernel
        vkernel = ['gauss','square','custom']
        if kernel is None:

            tkernel = 0

        else:

            tkernel = -1

            for ii,key in zip(range(len(vkernel)),vkernel):

                if kernel.lower() in key:
                    tkernel = ii
                    break

            if tkernel < 0:
                self.__error('Type of kernel not recognized')
                return -1

            if tkernel == 0 or tkernel == 1:

                # par
                if par is None:

                    self.__error('par is required for kernel ' + \
                                 vkernel[tkernel])
                    return -1

                else:

                    ipar = par

                    if not isinstance(ipar, float):
                        self.__error('par must be a float number')
                        return -1

                l_PSF = copy.deepcopy(axis)

            elif tkernel == 2:

                # kernel_file
                if kernel_file is None:

                    self.__error('kernel_file is required for ' + \
                                 'kernel ' + vkernel[tkernel])
                    return -1

                else:

                    ikernel_file = kernel_file

                    if not isinstance(ikernel_file, str):
                        self.__error('kernel_file must be a string')
                        return -1

                # Read Kernel
                try:
                    f_kernel = open(ikernel_file, 'rb')
                except IOError:
                    self.__error('kernel_file not found')
                    return -1
                except:
                    msg = 'problem opening kernel file\n' + \
                          sys.exc_info()[0] + '\n' + \
                          sys.exc_info()[1]
                    self.__error(msg)
                    return -1

                bytes = f_kernel.read(4)
                nl = int(struct.unpack('i', bytes)[0])
                bytes = f_kernel.read(8*nl)
                l_PSF = np.array(struct.unpack('d'*nl, bytes), \
                                 dtype=np.float64)
                bytes = f_kernel.read(8*nl)
                PSF = np.array(struct.unpack('d'*nl, bytes), \
                               dtype=np.float64)
                Pmin = np.min(l_PSF)
                Pmax = np.max(l_PSF)
                f_kernel.close()

        # Center
        if center is None:

            if tkernel == 0 or tkernel == 1:

                self.__error('center is required for ' + \
                             'kernel ' + vkernel[tkernel])
                return -1

        else:

            icenter = center

            if not isinstance(icenter, float):
                self.__error('center must be float type')
                return -1


        # Check if equidistant grid
        ds = iaxis[1:] - iaxis[0:-1]

        if np.min(ds) <= 0.:
            self.__error('axis must have only unique values')
            return -1

        if not iequi:
            iequi = True
            for ii in range(0,len(ds)-1):
                for jj in range(ii+1,len(ds)):
                    if (ds[ii] - ds[jj])/(ds[ii] + ds[jj]) > 1e-8:
                        iequi = False
                        break
                if not iequi:
                    break


        #
        # Manage axis
        #

        if iextend:

            axis0 = min([np.min(iaxis),np.min(l_PSF)])
            axis1 = max([np.max(iaxis),np.max(l_PSF)])

        else:

            axis0 = np.min(iaxis)
            axis1 = np.max(iaxis)

        # If custom, fit to its axis, because no filter outside
        if tkernel == 2:

            if axis0 < Pmin:
                axis0 = Pmin
            if axis1 > Pmax:
                axis1 = Pmax

        if iequi:
            dss = ds[0]
        else:
            dss = np.amin(ds)

        NNi = int((axis1 - axis0)/dss + 1)
        axisi = np.linspace(0,NNi-1,num=NNi,dtype=iaxis.dtype)
        axisi *= dss
        axisi += axis0


        # Interpolate
        if axis0 != np.min(iaxis) and axis1 != np.max(iaxis) and \
           NNi != iaxis.size:

            idata = np.interp(axisi, iaxis, idata, \
                              left=idata[0], right=idata[-1])


        #
        # Filter
        #


        # Gauss
        if tkernel == 0:

            PSF = np.exp(-0.5*(axisi - icenter)* \
                              (axisi - icenter)/ipar/ipar)

        # Square
        elif tkernel == 1:

            PSF = np.zeros((axisi.size))
            PSF = np.where(np.absolute(axisi - icenter) <= par, \
                           PSF + 1.0, PSF)

        # Custom MTF
        elif tkernel == 2:

            PSF = np.interp(axisi, l_PSF, PSF, left=0.0, \
                            right=0.0)

        idata *= PSF
        idata = np.trapz(idata,x=axisi)

        if inorm:
            idata /= np.trapz(PSF,x=axisi)

        return idata

