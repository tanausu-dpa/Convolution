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
#  11/11/2025 - V0.1.3 - Added the option to input more dimensions   #
#                        than the ones to convolve (TdPA)            #
#                      - Several improvements like using the pyfftw  #
#                        library (TdPA)                              #
#                                                                    #
#  04/04/2024 - V0.1.2 - Added m2nm() method (TdPA)                  #
#                                                                    #
#  04/05/2023 - V0.1.1 - Bugfix: The PDF for gauss and square for    #
#                        the 1D filter was using the wrong sigma     #
#                        variable, the input instead of the local    #
#                        one (TdPA)                                  #
#                                                                    #
#  27/07/2020 - V0.1.0 - Changed construction of input axis when not #
#                        equidistant because there could be a        #
#                        problem with the boundaries (TdPA)          #
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
import copy as lcopy
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
try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fftw
    pyfftw.interfaces.cache.enable()
    lpyfftw = True
except ModuleNotFoundError:
    lpyfftw = False
except ImportError:
    lpyfftw = False
except:
    raise

######################################################################
######################################################################
######################################################################
######################################################################

class degrade_class():
    ''' Class with functions to convolve and integrate
    '''

    def __init__(self,nthread=8):
        ''' Initialize class
        '''

        # Useful constants
        self.__cm2s = 1.3787950676165365e-8 
        self.__fwhm2sig = 4.24660900144009534e-1
        self.__sqrt2pi = 2.50662827463100024e0
        self.__deg2ra = 1.74532925199432955e-2
        self.__ra2deg = 5.72957795130823229e1
        self.__ck = False

        # pyfftw and number of threads
        self.__pyfftw = lpyfftw
        self.__nthread = nthread

        # If not pyfftw
        if not self.__pyfftw:
            print(' # The module pyfftw was not found. The class ' + \
                  'will use numpy, but it is very recommended to ' + \
                  'have pyfftw available for large data')

        # Valid kernels in 1D and 2D
        self.__vkernel1D = ['gaussian','square','custom']
        self.__vkernelFi = ['gaussian','square','custom']
        self.__vkernel2D = ['gaussian','square','airy','seeing']

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

    def __interpolation(self,idata,x0,x1):
        ''' Interpolation function for convolve1D
        '''
        odata = np.empty((idata.shape[0],x1.size))
        for icol in range(idata.shape[0]):
            odata[icol,:] = np.interp(x1,x0,idata[icol,:])
        return odata

######################################################################
######################################################################

    def __interpolation2D(self,idata,ins,xo,yo):
        ''' Interpolation function for convolve2D
        '''
        odata = np.empty((idata.shape[0],xo.shape[0],xo.shape[1]))
        for icol in range(idata.shape[0]):
            r = interpolate.RegularGridInterpolator(ins,idata[icol,:])
            odata[icol,:,:] = r((yo,xo))
        return odata

######################################################################
######################################################################

    def __setdomain(self,iaxis,iequi,iextend,iperiod,dis,NN):
        ''' Check and manage input axis and flag
        '''

        # Check if equidistant grid
        ds = iaxis[1:] - iaxis[0:-1]

        # If not unique values
        if np.min(ds) <= 0.:
            self.__error('axis must have only unique values and ' + \
                         'must increase monotonically')
            return None,None,None,None,None

        # If the input is not expected equidistant
        if not iequi:

            # Flag as it were
            iequi = True

            # For each distance
            for ii,dsi in enumerate(ds[:-1]):
                # For every other distance
                for dsj in ds[ii+1:]:
                    # If different
                    if np.absolute(dsi - dsj)/(dsi + dsj) > 1e-8:
                        # Flag not equidistant and finish
                        iequi = False
                        break
                # If flagged, we finished
                if not iequi:
                    break

        #
        # Manage axis
        #

        # If equidistant
        if iequi:
            NNi = NN
            dss = ds[0]
            axisi = np.linspace(0,NNi-1,num=NNi, \
                                endpoint=True,dtype=iaxis.dtype)
        else:
            dss = np.amin(ds)
            NNi = int(np.ceil((iaxis[-1] - iaxis[0])/dss) + 1)
            axisi = np.linspace(0,NNi-1,num=NNi, \
                                endpoint=True,dtype=iaxis.dtype)
            dss = (iaxis[-1] - iaxis[0])/float(NNi-1)

        # Domain sizes
        DD = axisi[-1] - axisi[0]
        DD *= dss

        #
        # Check padding
        #

        NNe = 0

        # If extending the borders as constants
        if iextend:

            NNe = int(dis/dss + 1)

        elif iperiod:

           if dis > DD:
                NNe = int((dis - DD)/dss + 1)

        # Return
        return axisi,dss,NNe,NNi,iperiod and dis > DD

######################################################################
######################################################################

    def cm2sec(self, x):
        ''' Transform argument from centimeter to arcsec
        '''
        return x*self.__cm2s

######################################################################
######################################################################

    def m2sec(self, x):
        ''' Transform argument from meter to arcsec
        '''
        return x*self.__cm2s*1e2

######################################################################
######################################################################

    def km2sec(self, x):
        ''' Transform argument from kilometer to arcsec
        '''
        return x*self.__cm2s*1e5

######################################################################
######################################################################

    def Mm2sec(self, x):
        ''' Transform argument from Megameter to arcsec
        '''

        return x*self.__cm2s*1e8

######################################################################
######################################################################

    def sec2cm(self, x):
        ''' Transform argument from arcsec to centimeter
        '''
        return x/self.__cm2s

######################################################################
######################################################################

    def sec2m(self, x):
        ''' Transform argument from arcsec to meter
        '''
        return x*1e-2/self.__cm2s

######################################################################
######################################################################

    def fwhm2sig(self, x):
        ''' Transform full width at half maximum to sigma in Gaussian
        '''
        return x*self.__fwhm2sig

######################################################################
######################################################################

    def deg2ra(self, x):
        ''' Transform argument from degrees to radians
        '''
        return x*self.__deg2ra

######################################################################
######################################################################

    def sec2ra(self, x):
        ''' Transform argument from seconds to radians
        '''
        return x*self.__deg2ra/3600.

######################################################################
######################################################################

    def ra2deg(self, x):
        ''' Transform argument form radian to degrees
        '''
        return x*self.__ra2deg

######################################################################
######################################################################

    def ra2sec(self, x):
        ''' Transform argument from radian to seconds
        '''
        return x*self.__ra2deg*3600.

######################################################################
######################################################################

    def m2nm(self, x):
        ''' Transform argument from meter to nanometer
        '''
        return x*1e9

######################################################################
######################################################################

    def nm2m(self, x):
        ''' Transform argument from nanometer to meter
        '''
        return x*1e-9

######################################################################
######################################################################

    def Ang2m(self, x):
        ''' Transform argument from angstrom to meter
        '''
        return x*1e-10

######################################################################
######################################################################

    def m2Ang(self, x):
        ''' Transform argument from meter to angstrom
        '''
        return x*1e10

######################################################################
######################################################################

    def mAng2m(self, x):
        ''' Transform argument from miliangstrom to meter
        '''
        return x*1e-13

######################################################################
######################################################################

    def setcustom1Dkernel(self, kernel_file):
        ''' Load the class with a custom kernel from a file
              Format:
                   (int)  Size of axis (n)
              n*(double)  Axis
              n*(double)  Kernel
        '''

        # kernel_file
        if kernel_file is None:

            self.__error('kernel_file is required to ' + \
                         'set up a kernel')
            return False

        else:

            ikernel_file = kernel_file

            if not isinstance(ikernel_file, str):
                self.__error('kernel_file must be a string')
                return False

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
            return False

        bytes = f_kernel.read(4)
        self.__ck_nl = int(struct.unpack('i', bytes)[0])
        nl = self.__ck_nl
        bytes = f_kernel.read(8*nl)
        self.__ck_l_PSF = np.array(struct.unpack('d'*nl, bytes), \
                                   dtype=np.float64)
        bytes = f_kernel.read(8*nl)
        self.__ck_PSF = np.array(struct.unpack('d'*nl, bytes), \
                                   dtype=np.float64)
        f_kernel.close()
        self.__ck = True
        return True

######################################################################
######################################################################

    def unsetcustom1Dkernel(self):
        ''' Unload the class of a pre-set custom kernel
        '''
        del self.__ck_nl
        del self.__ck_l_PSF
        del self.__ck_PSF
        self.__ck = False
        return True

######################################################################
######################################################################

    def __convol2D_single(self,idata,xaxisi,yaxisi,dsx,dsy,NXe,NYe, \
                          NXi,NYi,ixequi,iyequi,iextend,iwrap, \
                          tkernel,ixpar,iypar,l0,iD,r0,ixaxis,iyaxis):
        ''' Applies a 2D convolution
        '''

        # Interpolate
        if not ixequi or not iyequi:

            # If interpolating both dimensions
            if (not ixequi) and (not iyequi):

                # Create mesh
                xx, yy = np.meshgrid(xaxisi*dsx, yaxisi*dsy, \
                                     indexing='xy')

            # Only interpolating in X
            elif not ixequi:

                # Create mesh
                xx, yy = np.meshgrid(xaxisi*dsx, iyaxis, \
                                     indexing='xy')

            # Only interpolating in Y
            elif not iyequi:

                # Create mesh
                xx, yy = np.meshgrid(ixaxis, yaxisi*dsy, \
                                     indexing='xy')

            # Create interpolation function
            r = interpolate.RegularGridInterpolator((iyaxis,ixaxis), \
                                                    idata)
            # Interpolate into regular grid
            idata = r((yy,xx))


        #
        # Check padding
        #

        # If extending the borders as constants
        if iextend:
            idata = np.pad(idata,((NYe,NYe),(NXe,NXe)),'edge')

        # If wrapping
        elif iwrap:
            idata = np.pad(idata,((NYe,NYe),(NXe,NXe)),'wrap')

        # Shapes
        NYc, NXc = idata.shape

        # Build frequency axes
        u = np.fft.fftfreq(NXc, d=dsx)
        v = np.fft.fftfreq(NYc, d=dsy)

        # Build mesh
        uu, vv = np.meshgrid(u,v,indexing='xy')

        #
        # Build kernel
        #

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

            uv = self.ra2sec(l0*np.sqrt(uu*uu + vv*vv)/iD)

            uv = np.clip(uv, a_min=-1, a_max=1.)

            MTF = 2.*(np.arccos(uv) - uv*np.sqrt(1.-uv*uv))/ \
                  (np.pi*dsx*dsy)

        # MTF Fried (1966), Long-Exposure approximation
        elif tkernel == 3:

            uv = self.ra2sec(l0*np.sqrt(uu*uu + vv*vv)/iD)

            uv = np.clip(uv, a_min=-1., a_max=1.)

            MTF = np.exp(-3.44*np.power(uv*iD/r0,5./3.))* \
                  2.*(np.arccos(uv) - uv*np.sqrt(1.-uv*uv))/ \
                  (np.pi*dsx*dsy)

        # If library
        if self.__pyfftw:

            # Transform
            idata = fftw.fft2(idata)

            # Convolve
            idata *= MTF

            # Transform back
            idata = fftw.ifft2(idata).real*dsx*dsy

        # Only numpy
        else:

            # Transform
            idata = np.fft.fft2(idata)

            # Convolve
            idata *= MTF

            # Transform back
            idata = np.fft.ifft2(idata).real*dsx*dsy

        # If padding X
        if NXe > 0:
            idata = idata[:,:NXe+NXi]
            idata = idata[:,NXe:]

        # If padding Y
        if NYe > 0:
            idata = idata[:NYe+NYi,:]
            idata = idata[NYe:,:]

        # Interpolation
        if not ixequi or not iyequi:

            # Get mesh
            xx, yy = np.meshgrid(ixaxis, iyaxis, indexing='xy')

            # None are the same
            if (not ixequi) and (not iyequi):

                # Original axes
                dims = (yaxisi*dsy, xaxisi*dsx)

            # Only X differs
            elif not ixequi:

                # Original axes
                dims = (iyaxis, xaxisi*dsx)

            # Only Y differs
            elif not iyequi:

                # Original axes
                dims = (yaxisi*dsy, ixaxis)

            # Create interpolator
            r = interpolate.RegularGridInterpolator(dims, idata)

            # Interpolate
            idata = r((yy,xx))

        # Return
        return idata

######################################################################
######################################################################

    def __convol2D_many(self,idata,ixdim,iydim,xaxisi,yaxisi, \
                        dsx,dsy,NXe,NYe,NXi,NYi,ixequi,iyequi, \
                        iextend,iwrap,tkernel,ixpar,iypar, \
                        l0,iD,r0,ixaxis,iyaxis):
        ''' Applies a 2D convolution
        '''

        # Original dimensions
        nx, ny = idata.shape[ixdim], idata.shape[iydim]
        shape_info = idata.shape

        # Number of dimensions
        ndim = len(shape_info)

        # Determine permutation order — move selected axes to the end
        other_axes = [i for i in range(ndim) \
                      if i not in (ixdim, iydim)]
        perm = other_axes + [iydim, ixdim]

        # Transpose to make (others, dimx, dimy)
        idata = np.transpose(idata, axes=perm)

        # Compute collapsed shape
        n_other = int(np.prod([idata.shape[:-2]]))

        # Reshape to (collapsed, ny, nx)
        idata = idata.reshape((n_other, ny, nx))

        # Save info to restore
        shape_info = (shape_info, perm, other_axes, ixdim, iydim)

        # Interpolate
        if not ixequi or not iyequi:

            # If interpolating both dimensions
            if (not ixequi) and (not iyequi):

                # Create mesh
                xx, yy = np.meshgrid(xaxisi*dsx, yaxisi*dsy, \
                                     indexing='xy')

            # Only interpolating in X
            elif not ixequi:

                # Create mesh
                xx, yy = np.meshgrid(xaxisi*dsx, iyaxis, \
                                     indexing='xy')

            # Only interpolating in Y
            elif not iyequi:

                # Create mesh
                xx, yy = np.meshgrid(ixaxis, yaxisi*dsy, \
                                     indexing='xy')

            # Interpolate
            idata = self.__interpolation2D(idata,(iyaxis,ixaxis), \
                                           xx,yy)

        #
        # Check padding
        #

        # If extending the borders as constants
        if iextend:
            idata = np.pad(idata,((0,0),(NYe,NYe),(NXe,NXe)),'edge')

        # If wrapping
        elif iwrap:
            idata = np.pad(idata,((0,0),(NYe,NYe),(NXe,NXe)),'wrap')

        # Shapes
        NYc, NXc = idata.shape[-2:]

        # Build frequency axes
        u = np.fft.fftfreq(NXc, d=dsx)
        v = np.fft.fftfreq(NYc, d=dsy)

        # Build mesh
        uu, vv = np.meshgrid(u,v,indexing='xy')

        # Create aligned input/output buffers
        if self.__pyfftw:

            # Shape
            shape = idata.shape[-2:]

            # Create aligned arrays for input and output
            infftw = pyfftw.empty_aligned(shape, dtype='complex64')
            oufftw = pyfftw.empty_aligned(shape, dtype='complex64')

            # Create plan
            fft_forward = pyfftw.FFTW(infftw,oufftw,axes=(0,1), \
                                      direction='FFTW_FORWARD', \
                                      threads=self.__nthread)
            fft_inverse = pyfftw.FFTW(oufftw,infftw,axes=(0,1), \
                                      direction='FFTW_BACKWARD', \
                                      threads=self.__nthread)

        #
        # Build kernel
        #

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

            uv = self.ra2sec(l0*np.sqrt(uu*uu + vv*vv)/iD)

            uv = np.clip(uv, a_min=-1, a_max=1.)

            MTF = 2.*(np.arccos(uv) - uv*np.sqrt(1.-uv*uv))/ \
                  (np.pi*dsx*dsy)

        # MTF Fried (1966), Long-Exposure approximation
        elif tkernel == 3:

            uv = self.ra2sec(l0*np.sqrt(uu*uu + vv*vv)/iD)

            uv = np.clip(uv, a_min=-1., a_max=1.)

            MTF = np.exp(-3.44*np.power(uv*iD/r0,5./3.))* \
                  2.*(np.arccos(uv) - uv*np.sqrt(1.-uv*uv))/ \
                  (np.pi*dsx*dsy)

        # Number of columns
        ncol = idata.shape[0]

        # For each column
        for icol in range(ncol):

            # If using pyfftw
            if self.__pyfftw:

                # Fill input
                infftw[:] = idata[icol,:,:]

                # Transform
                fft_forward()

                # Convolve
                oufftw *= MTF

                # Transform back
                fft_inverse()

                # Back
                idata[icol,:,:] = np.real(infftw)*dsx*dsy

            # Only numpy
            else:

                # Transform
                fft = np.fft.fft2(idata[icol,:,:])

                # Convolve
                fft *= MTF

                # Transform back
                idata[icol,:,:] = np.fft.ifft2(fft).real*dsx*dsy

        # If padding X
        if NXe > 0:
            idata = idata[:,:,:NXe+NXi]
            idata = idata[:,:,NXe:]

        # If padding Y
        if NYe > 0:
            idata = idata[:,:NYe+NYi,:]
            idata = idata[:,NYe:,:]

        # Interpolation
        if not ixequi or not iyequi:

            # Get mesh
            xx, yy = np.meshgrid(ixaxis, iyaxis, indexing='xy')

            # None are the same
            if (not ixequi) and (not iyequi):

                # Original axes
                dims = (yaxisi*dsy, xaxisi*dsx)

            # Only X differs
            elif not ixequi:

                # Original axes
                dims = (iyaxis, xaxisi*dsx)

            # Only Y differs
            elif not iyequi:

                # Original axes
                dims = (yaxisi*dsy, ixaxis)

            # Interpolate
            idata = self.__interpolation2D(idata,dims,xx,yy)

        #
        # Restore shape
        #

        # New shape
        new_shape = [shape_info[0][i] for i in shape_info[2]] + \
                    [idata.shape[-2],idata.shape[-1]]
        idata = idata.reshape(new_shape)

        # Permutations
        idata = np.transpose(idata, axes=np.argsort(shape_info[1]))

        # Return
        return idata

######################################################################
######################################################################

    def convol2D(self, data, xdim=None, ydim=None, xaxis=None, \
                 yaxis=None, xpar=None, ypar=None, xperiod=None, \
                 yperiod=None, xextend=None, yextend=None, \
                 xdir=None, ydir=None, kernel=None, r0=None, D=None, \
                 l0=None, xequi=None, yequi=None, copy=None):
        ''' Applies a 2D convolution. Arguments:
               data: Array to convolve. Needs to be a numpy array
                     or convertible into one, as well as a float
                     variant. It can have more than two-dimensions
                     if specifying xdim and ydim. Default ordering
                     if (Y,X), but can be changed with xdim and ydim
               xdim: Integer with the dimension of data corresponding
                     to the X axis
               ydim: Integer with the dimension of data corresponding
                     to the Y axis
              xaxis: Array with the X axis. If not specified, an
                     axis of indexes is built. Must be a float variant
              yaxis: Array with the Y axis. If not specified, an
                     axis of indexes is built. Must be a float variant
               xpar: Convolution parameter for the Gaussian or
                     Square kernels along the X axis. Must be a float
                     variant
               ypar: Convolution parameter for the Gaussian or
                     Square kernels along the Y axis. Must be a float
                     variant
            xperiod: If considering periodicity along the X axis. Must
                     be bool
            yperiod: If considering periodicity along the Y axis. Must
                     be bool
            xextend: If extending the X axis beyond boundaries.
                     Exclusive with xperiod. Must be bool
            yextend: If extending the Y axis beyond boundaries.
                     Exclusive with yperiod. Must be bool
               xdir: How much to extend the boundary if xextend. Must
                     be a float variant
               ydir: How much to extend the boundary if yextend. Must
                     be a float variant
             kernel: Type of kernel. Allowed: Gaussian, Square,
                     Airy, and seeing
                 r0: Seeing parameter for the seeing kernel. Must be
                     a float variant
                  D: Diameter for the Airy or seeing kernel. Must be
                     a float variant
                 l0: Wavelength for the Airy or seeing kernel. Must be
                     a float variant
              xequi: If the input X axis can be assummed equidistant.
                     Must be a bool
              yequi: If the input Y axis can be assummed equidistant.
                     Must be a bool
               copy: If must create a copy of the input data. Must be
                     a bool. Default is no copying
        '''

        #
        # Manage arguments
        #

        #
        # Copy
        #

        # If undefined
        if copy is None:

            # No copy
            icopy = False

        # Defined
        else:

            # Get copy
            icopy = copy

            # No bool
            if not isinstance(icopy,bool):
                self.__error('copy must be bool type')
                return -1

        #
        # Data
        #

        # Try to create a copy
        try:

            # If copy
            if icopy:

                # Get copy
                idata = lcopy.deepcopy(data)

            # No copy
            else:

                # Point to data
                idata = data

            # Check data type
            if 'float' not in str(data.dtype):
                self.__error('data must be float type')

        # The input was not a numpy array
        except AttributeError:

            # Try converting into array
            try:

                # If copy
                if icopy:

                    # First copy the input
                    idata = lcopy.deepcopy(data)

                # If not
                else:

                    # Point
                    idata = data

                # Now convert into array
                idata = np.array(idata)

            # Could not convert into array
            except:
                self.__error('unexpected error while making ' + \
                             'an array from the data')
                return -1

        # There was an error trying to copy
        except:
            self.__error('unexpected error while checking ' + \
                         'the data')
            return -1

        # Check data type
        if 'float' not in str(idata.dtype):
            self.__error('data must be float type')
            return -1

        # If the input data is not 2D
        if len(data.shape) != 2:

            # Check if dimensions were specified
            if xdim is None or ydim is None:
                self.__error('the variables xdim and ydim, ' + \
                             'indicating the dimension to ' + \
                             'convolve, must be an integer ' + \
                             'if the data is not two dimensional')
                return -1

            # For each dimension variable
            for i,a in zip([xdim,ydim],['x','y']):

                # Check that dimension is an integer
                if not isinstance(i,int):
                    self.__error(f'the variable {a}dim, ' + \
                                 f'indicating the dimension {a} ' + \
                                 'to convolve, ' + \
                                 'must be an integer if the data ' + \
                                 ' is not two dimensional')
                    return -1

                # Check dimension lower limit
                if i < 0:
                    self.__error(f'the dimension {a} to convolve ' + \
                                 'cannot possibly be below 0')
                    return -1

                # Check dimension upper limit
                if i >= len(idata.shape):
                    self.__error(f'the dimension {a}to convolve ' + \
                                 'cannot possibly be larger ' + \
                                 'or equal to the number of ' + \
                                 'dimensions in the array')
                    return -1

            # Flag that it is not 1D
            single = False
            ixdim = xdim
            iydim = ydim

        # Data is 2D
        else:

            # Flag single column
            single = True

            # If no dimension specified
            if xdim is None and ydim is None:

                # Assume image order
                ixdim = 1
                iydim = 0

            # At least one specified
            else:

                # X indicated
                if xdim is not None:

                    # Check that dimension is an integer
                    if not isinstance(xdim,int):
                        self.__error('the variable xdim, ' + \
                                     'indicating the dimension x ' + \
                                     'to convolve, ' + \
                                     'must be an integer if ' + \
                                     'specified')
                        return -1

                    # Check dimension lower limit
                    if xdim < 0:
                        self.__error('the dimension x to convolve ' +\
                                     'cannot possibly be below 0')
                        return -1

                    # Check dimension upper limit
                    if xdim >= len(idata.shape):
                        self.__error('the dimension x to convolve ' +\
                                     'cannot possibly be larger ' + \
                                     'or equal to the number of ' + \
                                     'dimensions in the array')
                        return -1

                # Y indicated
                if ydim is not None:

                    # Check that dimension is an integer
                    if not isinstance(ydim,int):
                        self.__error('the variable ydim, ' + \
                                     'indicating the dimension y ' + \
                                     'to convolve, ' + \
                                     'must be an integer if ' + \
                                     'specified')
                        return -1

                    # Check dimension lower limit
                    if ydim < 0:
                        self.__error('the dimension y to convolve ' +\
                                     'cannot possibly be below 0')
                        return -1

                    # Check dimension upper limit
                    if ydim >= len(idata.shape):
                        self.__error('the dimension y to convolve ' +\
                                     'cannot possibly be larger ' + \
                                     'or equal to the number of ' + \
                                     'dimensions in the array')
                        return -1

                # Number of dimensions
                ndim = idata.ndim

                # Both indicated
                if xdim is not None and ydim is not None:

                    # Copy
                    ixdim = xdim % ndim
                    iydim = ydim % ndim

                # Only X
                elif xdim is not None:

                    # Copy and assume
                    ixdim = xdim % ndim
                    iydim = 1 - ixdim

                # Only Y
                elif ydim is not None:

                    # Copy and assume
                    iydim = ydim % ndim
                    ixdim = 1 - iydim

                # Weird
                else:
                    self.__error('something wrong happened ' + \
                                 'when dealing with what is ' + \
                                 'x and what is y in the data')
                    return -1

                # Check not the same
                if ixdim == iydim:
                    self.__error('the dimensions x and y to ' + \
                                 'convolve cannot possibly ' + \
                                 'be the same')
                    return -1

        # Save relevant dimension sizes
        NX = idata.shape[ixdim]
        NY = idata.shape[iydim]

        # Check if sizes too small
        if NX < 2:
            self.__error('X size must be larger than 1')
            return -1
        if NY < 2:
            self.__error('Y size must be larger than 1')
            return -1

        #
        # Coordinates
        #

        #
        # X
        # If not specified
        if xaxis is None:

            # Create equidistant axis and flag it
            ixaxis = np.linspace(0.,1.*(NX-1),num=NX,endpoint=True)
            ixequi = False

        # Axis in input
        else:

            # Initialize ixequi as False, we read the input later
            ixequi = False 

            # Try to copy and treat as numpy array
            try:

                # Point
                ixaxis = xaxis

                # Check this here just to ensure numpy array
                if len(ixaxis.shape) != 1:
                    self.__error('xaxis must be 1D array')
                    return -1

            # If data was not numpy array
            except AttributeError:

                # Try to make it numpy array
                try:

                    # First copy
                    ixaxis = lcopy.deepcopy(xaxis)

                    # Now create array
                    ixaxis = np.array(ixaxis)

                # Could not make array
                except:
                    self.__error('unexpected error while making ' + \
                                 'an array from xaxis')
                    return -1

            # Could not copy
            except:
                self.__error('unexpected error while checking ' + \
                             'xaxis')
                return -1

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

        #
        # Y
        # If not specified
        if yaxis is None:

            # Create equidistant axis and flag it
            iyaxis = np.linspace(0.,1.*(NY-1),num=NY,endpoint=True)
            iyequi = False

        # Axis in input
        else:

            # Initialize iyequi as False, we read the input later
            iyequi = False 

            # Try to copy and treat as numpy array
            try:

                # Point
                iyaxis = yaxis

                # Check this here just to ensure numpy array
                if len(iyaxis.shape) != 1:
                    self.__error('yaxis must be 1D array')
                    return -1

            # If data was not numpy array
            except AttributeError:

                # Try to make it numpy array
                try:

                    # First copy
                    iyaxis = lcopy.deepcopy(yaxis)

                    # Now create array
                    iyaxis = np.array(iyaxis)

                # Could not make array
                except:
                    self.__error('unexpected error while making ' + \
                                 'an array from yaxis')
                    return -1

            # Could not copy
            except:
                self.__error('unexpected error while checking ' + \
                             'yaxis')
                return -1

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


        #
        # Periodicity
        #

        #
        # X
        # If not defined
        if xperiod is None:

            # Assume not periodic
            ixperiod = False

        # Defined
        else:

            # Copy input
            ixperiod = xperiod

            # Check it is a bool
            if not isinstance(ixperiod, bool):
                self.__error('xperiod must be bool')
                return -1


        #
        # Y
        # If not defined
        if yperiod is None:

            # Assume not periodic
            iyperiod = False

        # Defined
        else:

            # Copy input
            iyperiod = yperiod

            # Check it is a bool
            if not isinstance(iyperiod, bool):
                self.__error('yperiod must be bool')
                return -1


        #
        # Extend beyond limits
        #

        #
        # X
        # If not defined
        if xextend is None:

            # Don't extend
            ixextend = False

        # If there is an input
        else:

            # Copy
            ixextend = xextend

            # Check it is a bool
            if not isinstance(ixextend, bool):
                self.__error('xextend must be bool')
                return -1

            # Check exclusivity
            if ixperiod and ixextend:
                self.__error('xextend and xperiod cannot ' + \
                             'be True at the same time')
                return -1


        #
        # Y
        # If not defined
        if yextend is None:

            # Don't extend
            iyextend = False

        # If there is an input
        else:

            # Copy
            iyextend = yextend

            # Check it is a bool
            if not isinstance(iyextend, bool):
                self.__error('yextend must be bool')
                return -1

            # Check exclusivity
            if iyperiod and iyextend:
                self.__error('yextend and yperiod cannot ' + \
                             'be True at the same time')
                return -1


        #
        # Equidistant axis
        #

        #
        # X
        # Only if there is input, because it is already
        # initialized, and if there was input axis
        if xequi is not None and xaxis is not None:

            # Copy input
            ixequi = xequi

            # Check if bool
            if not isinstance(ixequi, bool):
                self.__error('xequi must be bool')
                return -1

        #
        # Y
        # Only if there is input, because it is already
        # initialized, and if there was input axis
        if yequi is not None and yaxis is not None:

            # Copy input
            iyequi = yequi

            # Check if bool
            if not isinstance(iyequi, bool):
                self.__error('yequi must be bool')
                return -1


        #
        # How much to extend
        #

        #
        # X
        # If not indicated
        if xdir is None:

            # Default value of factor 5
            ixdir = 5

        # If there is input
        else:

            # Copy input
            ixdir = xdir

            # Check it is a float
            if not isinstance(ixdir, float):
                self.__error('xdir must be a float number')
                return -1

        #
        # Y
        # If not indicated
        if ydir is None:

            # Default value of factor 5
            iydir = 5

        # If there is input
        else:

            # Copy input
            iydir = ydir

            # Check it is a float
            if not isinstance(iydir, float):
                self.__error('ydir must be a float number')
                return -1


        #
        # Convolution kernel
        #

        # If no input
        if kernel is None:

            # Assume Gaussian
            tkernel = 0

        # Specified
        else:

            # Initialize ID
            tkernel = -1

            # Check all valid kernels
            for ii,key in enumerate(self.__vkernel2D):

                # If input in the list
                if kernel.lower() in key:

                    # Identify and leave
                    tkernel = ii
                    break

            # If not found
            if tkernel < 0:
                self.__error('Type of kernel not recognized')
                return -1

            # If Gaussian or square
            if tkernel == 0 or tkernel == 1:

                # Initialize others
                il0 = 0.
                iD = 0.
                ir0 = 0.

                #
                # Check parameter
                #

                #
                # X
                # If not defined
                if xpar is None:

                    # Error, because it is required
                    self.__error('xpar is required for kernel ' + \
                                 self.__vkernel2D[tkernel])
                    return -1

                # If there is input
                else:

                    # Copy parameter
                    ixpar = xpar

                    # Check it is float
                    if not isinstance(ixpar, float):
                        self.__error('xpar must be a float number')
                        return -1

                #
                # Y
                # If not defined
                if ypar is None:

                    # Error, because it is required
                    self.__error('ypar is required for kernel ' + \
                                 self.__vkernel2D[tkernel])
                    return -1

                # If there is input
                else:

                    # Copy parameter
                    iypar = ypar

                    # Check it is float
                    if not isinstance(iypar, float):
                        self.__error('ypar must be a float number')
                        return -1

            # If Airy or seeing
            if tkernel == 2 or tkernel == 3:

                #
                # Initialize others
                ixpar = 0.
                iypar = 0.

                #
                # Wavelength
                # If not defined
                if l0 is None:

                    # Error, because it is required
                    self.__error('l0 is required for kernel ' + \
                                 self.__vkernel2D[tkernel])
                    return -1

                # If there is input
                else:

                    # Copy parameter
                    il0 = l0

                    # Check it is float
                    if not isinstance(il0, float):
                        self.__error('l0 must be a float number')
                        return -1

                #
                # Diameter
                # If not defined
                if D is None:

                    # Error, because it is required
                    self.__error('D is required for kernel ' + \
                                 self.__vkernel2D[tkernel])
                    return -1

                # If there is input
                else:

                    # Copy parameter
                    iD = D

                    # Check it is float
                    if not isinstance(iD, float):
                        self.__error('D must be a float number')
                        return -1

                # If seeing
                if tkernel == 3:

                    #
                    # Coherence radius
                    # If not defined
                    if r0 is None:

                        # Error, because it is required
                        self.__error('r0 is required for kernel ' + \
                                     self.__vkernel2D[tkernel])
                        return -1

                    # If there is input
                    else:

                        # Copy parameter
                        ir0 = r0

                        # Check it is float
                        if not isinstance(ir0, float):
                            self.__error('r0 must be a float number')
                            return -1

        #
        # Get displacement sizes
        #

        # If gaussian, Airy, or seeing
        if tkernel == 0 or tkernel == 2 or tkernel == 3:

            # If there was input parameter for X
            if xpar is not None:

                # Scale by parameter
                xdis = ixdir*xpar

            # If there was not
            else:

                # Assume 1
                xdis = ixdir

            # If there was input parameter for Y
            if ypar is not None:

                # Scale by parameter
                ydis = iydir*ypar

            # If there was not
            else:

                # Assume 1
                ydis = iydir

        # If square
        elif tkernel == 1:

            # Scale by 5 always
            xdis = xpar*.5
            ydis = ypar*.5

        # Save origins
        xori = ixaxis[0]
        yori = iyaxis[0]

        # Shift coordinate axes to 0 origin
        ixaxis -= xori
        iyaxis -= yori

        # Manage X axis
        xaxisi,dsx,NXe,NXi,ixwrap = self.__setdomain(ixaxis,ixequi, \
                                                     ixextend, \
                                                     ixperiod, \
                                                     xdis,NX)
        # Check error
        if xaxisi is None: return -1

        # Manage Y axis
        yaxisi,dsy,NYe,NYi,iywrap = self.__setdomain(iyaxis,iyequi, \
                                                     iyextend, \
                                                     iyperiod, \
                                                     ydis,NY)
        # Check error
        if yaxisi is None: return -1

        # If the data was 2D
        if single:

            # If x was the first
            if ixdim == 0:
                idata = np.transpose(idata)

            # Convolution
            idata = self.__convol2D_single(idata,xaxisi,yaxisi, \
                                           dsx,dsy,NXe,NYe,NXi,NYi, \
                                           ixequi,iyequi, \
                                           ixextend or iyextend, \
                                           ixwrap or iywrap, \
                                           tkernel,ixpar,iypar, \
                                           l0,iD,r0, \
                                           ixaxis,iyaxis)

            # If x was the first
            if ixdim == 0:
                idata = np.transpose(idata)

        # Data was not 2D
        else:

            # Convolution
            idata = self.__convol2D_many(idata,ixdim,iydim, \
                                         xaxisi,yaxisi, \
                                         dsx,dsy,NXe,NYe,NXi,NYi, \
                                         ixequi,iyequi, \
                                         ixextend or iyextend, \
                                         ixwrap or iywrap, \
                                         tkernel,ixpar,iypar, \
                                         l0,iD,r0, \
                                         ixaxis,iyaxis)

        # Shift coordinate back
        ixaxis += xori
        iyaxis += yori

        # Return
        return idata

######################################################################
######################################################################

    def __convol1D_single(self,idata,axisi,dss,NNe,NNi,iequi, \
                          iextend,iwrap,tkernel,ipar,iPSF,l_PSF, \
                          iaxis):
        ''' Applies a 1D convolution/deconvolution
        '''

        # Interpolate
        if not iequi:
             idata = np.interp(axisi*dss, iaxis, idata)

        #
        # Check padding
        #

        # If extending the borders as constants
        if iextend:
            idata = np.pad(idata,((NNe,NNe)),'edge')

        # If wrapping
        elif iwrap:
            idata = np.pad(idata,((NNe,NNe)),'wrap')

        # Actual size
        NNc = idata.shape[0]

        # Build frequency axis
        u = np.fft.fftfreq(NNc, d=dss)

        #
        # Build kernel
        #

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
            ipar = np.trapz(iPSF,l_PSF)
            PSF = iPSF*dss/ipar

            # Get the sizes of axis and data
            Dl0 = axisi[-1]*dss
            Dl = (idata.size-1)*dss

            # Compute how much out for one side
            Dout = (l_PSF[-1] - Dl)*.5

            # Extended axis
            axisi_tmp = np.linspace(0, idata.size-1, \
                                    num=idata.size, \
                                    endpoint=True, \
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

        # If library
        if self.__pyfftw:

            # Transform
            idata = fftw.fft(idata)

            # Convolution
            idata *= MTF

            # Transform back
            idata = fftw.ifft(idata).real*ifact

        # Only numpy
        else:

            # Transform
            idata = np.fft.fft(idata)

            # Convolution
            idata *= MTF

            # Transform back
            idata = np.fft.ifft(idata).real*ifact

        # If padding
        if NNe > 0:
            idata = idata[:NNe+NNi]
            idata = idata[NNe:]

        # Interpolation
        if not iequi:
            return np.interp(iaxis,axisi*dss,idata)

        # Return
        return idata

######################################################################
######################################################################

    def __convol1D_many(self,idata,idim,axisi,dss,NNe,NNi,iequi, \
                        iextend,iwrap,tkernel,ipar,iPSF,l_PSF, \
                        iaxis):
        ''' Applies a 1D convolution/deconvolution
        '''

        # Transform
        idata = np.moveaxis(idata,idim,-1)
        oshape = idata.shape
        idata = idata.reshape(-1,idata.shape[-1])

        # Interpolation
        if not iequi:
            idata = self.__interpolation(idata,iaxis,axisi*dss)

        #
        # Check padding
        #

        # If extending the borders as constants
        if iextend:
            idata = np.pad(idata,((0,0),(NNe,NNe)),'edge')

        # If wrapping
        elif iwrap:
            idata = np.pad(idata,((0,0),(NNe,NNe)),'wrap')

        # Actual size
        NNc = idata.shape[1]

        # Build frequency axis
        u = np.fft.fftfreq(NNc, d=dss)

        # Create aligned input/output buffers
        if self.__pyfftw:

            # Create memory for pyfftw
            infftw = pyfftw.empty_aligned(NNc, dtype='complex128')
            oufftw = pyfftw.empty_aligned(NNc, dtype='complex128')

            # Create plans
            fft_forward = pyfftw.FFTW(infftw,oufftw, \
                                      direction='FFTW_FORWARD', \
                                      threads=self.__nthread)
            fft_inverse = pyfftw.FFTW(oufftw,infftw, \
                                      direction='FFTW_BACKWARD', \
                                      threads=self.__nthread)

        #
        # Build kernel
        #

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
            ipar = np.trapz(iPSF,l_PSF)
            PSF = iPSF*dss/ipar

            # Get the sizes of axis and data
            Dl0 = axisi[-1]*dss
            Dl = (idatum.size-1)*dss

            # Compute how much out for one side
            Dout = (l_PSF[-1] - Dl)*.5

            # Extended axis
            axisi_tmp = np.linspace(0,idata.shape[-1]-1, \
                                    num=idata.shape[-1], \
                                    endpoint=True, \
                                    dtype=axisi.dtype)

            # Shift axis
            axisi_tmp *= dss
            axisi_tmp += Dout

            # Inteprolate
            PSF = np.interp(axisi_tmp,l_PSF,PSF, \
                            left=0.,right=0.)

            # Free
            axisi_tmp = []

            MTF = np.fft.fft(PSF)
            MTF *= np.conj(MTF)
            MTF = np.sqrt(MTF)

            ifact = 1e0

        # Number of columns
        ncol = idata.shape[0]

        # For each column
        for icol in range(ncol):

            # If using pyfftw
            if self.__pyfftw:

                # Fill input
                infftw[:] = idata[icol,:]

                # Transform
                fft_forward()

                # Convolve
                oufftw *= MTF

                # Transform back
                fft_inverse()

                # Back
                idata[icol,:] = np.real(infftw)*ifact

            else:

                # Transform
                idatum = idata[icol,:]
                idatum = np.fft.fft(idata[icol,:])

                # Convolution
                idatum *= MTF

                # Back to lambda space
                idatum = np.fft.ifft(idatum).real*ifact

                # Save
                idata[icol,:] = idatum

        # If padding
        if NNe > 0:
            idata = idata[:,:NNe+NNi]
            idata = idata[:,NNe:]

        # Interpolation
        if not iequi:
            idata = self.__interpolation(idata,axisi*dss,iaxis)

        # Transform back
        idata = idata.reshape(oshape[:-1] + (oshape[-1],))
        idata = np.moveaxis(idata,-1,idim)

        # Return
        return idata

######################################################################
######################################################################

    def convol1D(self, data, dim=None, axis=None, par=None, \
                 period=None, extend=None, adir=None, kernel=None, \
                 equi=None, kernel_file=None, copy=None):
        ''' Applies a 1D convolution. Arguments:
               data: Array to convolve. Needs to be a numpy array
                     or convertible into one, as well as a float
                     variant. It can have more than one dimension
                     if specifying dim
                dim: Integer with the dimension of data to convolve
               axis: Array with the axis. If not specified, an
                     axis of indexes is built. Must be a float variant
                par: Convolution parameter for the Gaussian or
                     Square kernels. Must be a float variant
             period: If considering periodicity. Must be bool
             extend: If extending the axis beyond boundaries.
                     Exclusive with period. Must be bool
               adir: How much to extend the boundary if extend. Must
                     be a float variant
             kernel: Type of kernel. Allowed: Gaussian, Square,
                     and Custom
               equi: If the input axis can be assummed equidistant.
                     Must be a bool
        kernel_file: Path to the file containing the kernel for the
                     Custom case. Note that it can be preloaded with
                     degrade.setcustom1Dkernel(). Format:
                               (int)  Size of axis (n)
                          n*(double)  Axis
                          n*(double)  Kernel
               copy: If must create a copy of the input data. Must be
                     a bool. Default is no copying
        '''

        #
        # Manage arguments
        #

        #
        # Copy
        #

        # If undefined
        if copy is None:

            # No copy
            icopy = False

        # Defined
        else:

            # Get copy
            icopy = copy

            # No bool
            if not isinstance(icopy,bool):
                self.__error('copy must be bool type')
                return -1

        #
        # Data
        #

        # Try to create a copy
        try:

            # Copy data
            if icopy:
                idata = lcopy.deepcopy(data)
            # Point to data
            else:
                idata = data

            # Check data type
            if 'float' not in str(idata.dtype):
                self.__error('data must be float type')
                return -1

        # The input was not a numpy array
        except AttributeError:

            # Try converting into array
            try:

                # First copy the input
                if icopy:
                    idata = lcopy.deepcopy(data)
                # If point
                else:
                    idata = data

                # Now convert into array
                idata = np.array(idata)

            # Could not convert into array
            except:
                self.__error('unexpected error while making ' + \
                             'an array from the data')
                return -1

        # There was an error trying to copy
        except:
            self.__error('unexpected error while checking ' + \
                         'the data')
            return -1

        # Check data type
        if 'float' not in str(idata.dtype):
            self.__error('data must be float type')
            return -1

        # Flag single column and initialize dimension
        single = True
        idim = 0

        # If the input data is not 1D
        if len(idata.shape) != 1:

            # Check if dimension was specified
            if dim is None:
                self.__error('the variable dim, indicating ' + \
                             'the dimension to convolve, ' + \
                             'must be an integer if the data ' + \
                             ' is not one dimensional')
                return -1

            # Check that dimension is an integer
            if not isinstance(dim,int):
                self.__error('the variable dim, indicating ' + \
                             'the dimension to convolve, ' + \
                             'must be an integer if the data ' + \
                             ' is not one dimensional')
                return -1

            # Check dimension lower limit
            if dim < 0:
                self.__error('the dimension to convolve ' + \
                             'cannot possibly be below 0')
                return -1

            # Check dimension upper limit
            if dim >= len(idata.shape):
                self.__error('the dimension to convolve ' + \
                             'cannot possibly be larger ' + \
                             'or equal to the number of ' + \
                             'dimensions in the array')
                return -1

            # Flag that it is not 1D
            single = False
            idim = dim

        # Save relevant dimension size
        NN = idata.shape[idim]

        # Check if size is too small
        if NN < 2:
            self.__error('data size must be larger than 1')
            return -1

        #
        # Coordinates
        #

        # If not specified
        if axis is None:

            # Create equidistant axis and flag it
            iaxis = np.linspace(0.,1.*(NN-1),num=NN,endpoint=True)
            iequi = True

        # Axis in input
        else:

            # Initialize iequi as False, we read the input later
            iequi = False 

            # Try to copy and treat as numpy array
            try:

                # Make copy
                iaxis = axis

                # Check this here just to ensure numpy array
                if len(iaxis.shape) != 1:
                    self.__error('axis must be 1D array')
                    return -1

            # If data was not numpy array
            except AttributeError:

                # Try to make it numpy array
                try:

                    # First copy
                    iaxis = lcopy.deepcopy(axis)

                    # Now create array
                    iaxis = np.array(iaxis)

                # Could not make array
                except:
                    self.__error('unexpected error while making ' + \
                                 'an array from axis')
                    return -1

            # Could not copy
            except:
                self.__error('unexpected error while checking ' + \
                             'axis')
                return -1

        # Check coordinates are floats
        if 'float' not in str(iaxis.dtype):
            self.__error('axis must be float type')
            return -1

        # Check correct size
        if iaxis.shape[0] != NN:
            self.__error('dimension mismatch between ' + \
                         'data and axis')
            return -1

        #
        # Periodicity
        #

        # If not defined
        if period is None:

            # Assume not periodic
            iperiod = False

        # Defined
        else:

            # Copy input
            iperiod = period

            # Check it is a bool
            if not isinstance(iperiod, bool):
                self.__error('period must be bool')
                return -1


        #
        # Extend beyond limits
        #

        # If not indicated
        if extend is None:

            # Don't extend
            iextend = False

        # If there is an input
        else:

            # Copy
            iextend = extend

            # Check it is a bool
            if not isinstance(iextend, bool):
                self.__error('extend must be bool')
                return -1

            # Check exclusivity
            if iperiod and iextend:
                self.__error('extend and period cannot ' + \
                             'be True at the same time')
                return -1

        #
        # How much to extend
        #

        # If not indicated
        if adir is None:

            # Default value of factor 5
            idir = 5.0

        # If there is input
        else:

            # Copy input
            idir = adir

            # Check it is a float
            if not isinstance(idir, float):
                self.__error('adir must be a float number')
                return -1


        #
        # Equidistant axis
        #

        # Only if there is input, because it is already
        # initialized, and if there was input axis
        if equi is not None and axis is not None:

            # Copy input
            iequi = equi

            # Check if bool
            if not isinstance(iequi, bool):
                self.__error('equi must be a bool')
                return -1

        #
        # Convolution kernel
        #

        # If no input
        if kernel is None:

            # Assume Gaussian and put to zero customs
            tkernel = 0
            iPSF = None
            l_PSF = None

        # Specified
        else:

            # Initialize ID
            tkernel = -1

            # Check all valid kernels
            for ii,key in enumerate(self.__vkernel1D):

                # If input in the list
                if kernel.lower() in key:

                    # Identify and leave
                    tkernel = ii
                    break

            # If not found
            if tkernel < 0:
                self.__error('Type of kernel not recognized')
                return -1

            # If Gaussian or square
            if tkernel == 0 or tkernel == 1:

                #
                # Check parameter
                #

                # If not defined
                if par is None:

                    # Error, because it is required
                    self.__error('par is required for kernel ' + \
                                 self.__vkernel1D[tkernel])
                    return -1

                # If there is input
                else:

                    # Copy parameter
                    ipar = par

                    # Check it is float
                    if not isinstance(ipar, float):
                        self.__error('par must be a float number')
                        return -1

                # Initialize empty custom kernel variables
                iPSF = None
                l_PSF = None

            # If custom kernel
            elif tkernel == 2:

                # If the Kernel is set
                if self.__ck:

                    # Point to kernel data
                    nl = self.__ck_nl
                    l_PSF = self.__ck_l_PSF - self.__ck_l_PSF[0]
                    iPDF = self.__ck_PSF

                # Kernel if not set
                else:

                    #
                    # kernel_file
                    #

                    # If file not specified
                    if kernel_file is None:

                        # Error because it is required
                        self.__error('kernel_file is required ' + \
                                     'for kernel ' + \
                                     self.__vkernel1D[tkernel])
                        return -1

                    # There is input
                    else:

                        # Copy path
                        ikernel_file = kernel_file

                        # Check it is a string
                        if not isinstance(ikernel_file, str):
                            self.__error('kernel_file must be ' + \
                                         'a string')
                            return -1

                    # Try opening the file and handle errors
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

                    # Read kernel data
                    try:
                        nl = struct.unpack('i',f_kernel.read(4))[0]
                        il_PSF = np.array(struct.unpack('d'*nl, \
                                           f_kernel.read(8*nl)), \
                                          dtype=np.float64)
                        iPSF = np.array(struct.unpack('d'*nl, \
                                         f_kernel.read(8*nl)), \
                                        dtype=np.float64)
                    except:
                        f_kernel.close()
                        self.__error(f'problem reading kernel in ' + \
                                     f'{kernel_file}')
                        return -1
                    f_kernel.close()

                    # Shift axis to 0 origin
                    l_PSF = il_PSF - il_PSF[0]

        #
        # Get displacement sizes
        #

        # If Gaussian
        if tkernel == 0:

            # If there was input parameter
            if par is not None:

                # Scale by parameter
                dis = idir*par

            # There was no input parameter
            else:

                # Assume 1
                dis = idir

        # If square
        elif tkernel == 1:

            # Scale by 5 always
            dis = par*.5

        # Custom
        elif tkernel == 2:

            # Take last number, which must be size
            dis = l_PSF[-1]

        # Save origin
        ori = iaxis[0]

        # Shift coordinate axis to 0 origin
        iaxis -= ori

        # Manage axis
        axisi,dss,NNe,NNi,iwrap = self.__setdomain(iaxis,iequi, \
                                                   iextend, \
                                                   iperiod, \
                                                   dis,NN)

        # Check error
        if axisi is None: return -1

        # If the data was 1D
        if single:

            # Convolution
            idata = self.__convol1D_single(idata,axisi,dss,NNe,NNi, \
                                           iequi,iextend,iwrap, \
                                           tkernel,ipar,iPSF,l_PSF, \
                                           iaxis)
        # Data was not 1D
        else:

            # Convolution
            idata = self.__convol1D_many(idata,idim,axisi,dss, \
                                         NNe,NNi,iequi,iextend, \
                                         iwrap,tkernel,ipar, \
                                         iPSF,l_PSF,iaxis)

        # Shift coordinate back
        iaxis += ori

        # Return convolution
        return idata

######################################################################
######################################################################

    def filter1D(self, data, dim=None, axis=None, par=None, \
                 center=None, extend=None, kernel=None, equi=None, \
                 kernel_file=None, norm=None, copy=None):
        ''' Applies a 1D filter (integral). Arguments:
               data: Array to convolve. Needs to be a numpy array
                     or convertible into one, as well as a float
                     variant. It can have more than one dimension
                     if specifying dim
                dim: Integer with the dimension of data to convolve
               axis: Array with the axis. If not specified, an
                     axis of indexes is built. Must be a float variant
                par: Convolution parameter for the Gaussian or
                     Square kernels. Must be a float variant
             center: Center of the filter for the Gaussian or Square
                     kernels
             extend: If extending the axis beyond boundaries.
                     Exclusive with period. Must be bool
             kernel: Type of kernel. Allowed: Gaussian, Square,
                     and Custom
               equi: If the input axis can be assummed equidistant.
                     Must be a bool
        kernel_file: Path to the file containing the kernel for the
                     Custom case. Note that it can be preloaded with
                     degrade.setcustom1Dkernel(). Format:
                               (int)  Size of axis (n)
                          n*(double)  Axis
                          n*(double)  Kernel
               norm: If the integral is to be normalized by the
                     kernel integral. Must be a bool
               copy: If must create a copy of the input data. Must be
                     a bool. Default is no copying
        '''

        #
        # Manage arguments
        #

        #
        # Copy
        #

        # If undefined
        if copy is None:

            # No copy
            icopy = False

        # Defined
        else:

            # Get copy
            icopy = copy

            # No bool
            if not isinstance(icopy,bool):
                self.__error('copy must be bool type')
                return -1

        #
        # Data
        #

        # Try to create a copy
        try:

            # Copy data
            if icopy:
                idata = lcopy.deepcopy(data)
            # Point to data
            else:
                idata = data

            # Check data type
            if 'float' not in str(data.dtype):
                self.__error('data must be float type')

        # The input was not a numpy array
        except AttributeError:

            # Try converting into array
            try:

                # First copy the input
                if icopy:
                    idata = lcopy.deepcopy(data)
                # If point
                else:
                    idata = data

                # Now convert into array
                idata = np.array(idata)

            # Could not convert into array
            except:
                self.__error('unexpected error while making ' + \
                             'an array from the data')
                return -1

        # There was an error trying to copy
        except:
            self.__error('unexpected error while checking ' + \
                         'the data')
            return -1

        # Check data type
        if 'float' not in str(idata.dtype):
            self.__error('data must be float type')
            return -1

        # Flag single column and initialize dimension
        single = True
        idim = 0

        # If the input data is not 1D
        if len(idata.shape) != 1:

            # Check if dimension was specified
            if dim is None:
                self.__error('the variable dim, indicating ' + \
                             'the dimension to convolve, ' + \
                             'must be an integer if the data ' + \
                             ' is not one dimensional')
                return -1

            # Check that dimension is an integer
            if not isinstance(dim,int):
                self.__error('the variable dim, indicating ' + \
                             'the dimension to convolve, ' + \
                             'must be an integer if the data ' + \
                             ' is not one dimensional')
                return -1

            # Check dimension lower limit
            if dim < 0:
                self.__error('the dimension to convolve ' + \
                             'cannot possibly be below 0')
                return -1

            # Check dimension upper limit
            if dim >= len(idata.shape):
                self.__error('the dimension to convolve ' + \
                             'cannot possibly be larger ' + \
                             'or equal to the number of ' + \
                             'dimensions in the array')
                return -1

            # Flag that it is not 1D
            single = False
            idim = dim

        # Save relevant dimension size
        NN = idata.shape[idim]

        # Check if size is too small
        if NN < 2:
            self.__error('data size must be larger than 1')
            return -1

        #
        # Coordinates
        #

        # If not specified
        if axis is None:

            # Error because required
            self.__error('the abscissa is required')
            return -1

        # Axis in input
        else:

            # Initialize iequi as False, we read the input later
            iequi = False 

            # Try to copy and treat as numpy array
            try:

                # Make copy
                iaxis = axis

                # Check this here just to ensure numpy array
                if len(iaxis.shape) != 1:
                    self.__error('axis must be 1D array')
                    return -1

            # If data was not numpy array
            except AttributeError:

                # Try to make it numpy array
                try:

                    # First copy
                    iaxis = lcopy.deepcopy(axis)

                    # Now create array
                    iaxis = np.array(iaxis)

                # Could not make array
                except:
                    self.__error('unexpected error while making ' + \
                                 'an array from axis')
                    return -1

            # Could not copy
            except:
                self.__error('unexpected error while checking ' + \
                             'axis')
                return -1


        # Check coordinates are floats
        if 'float' not in str(iaxis.dtype):
            self.__error('axis must be float type')
            return -1

        # Check 1D
        if len(iaxis.shape) != 1:
            self.__error('axis must be 1D array')
            return -1

        # Check correct size
        if iaxis.shape[0] != NN:
            self.__error('dimension mismatch between ' + \
                         'data and axis')
            return -1


        #
        # Extend beyond limits
        #

        # If not indicated
        if extend is None:

            # Don't extend
            iextend = False

        # If there is an input
        else:

            # Copy
            iextend = extend

            # Check it is a bool
            if not isinstance(iextend, bool):
                self.__error('extend must be bool')
                return -1


        #
        # Equidistant axis
        #

        # Only if there is input
        if equi is not None:

            # Copy input
            iequi = equi

            # Check if bool
            if not isinstance(iequi, bool):
                self.__error('equi must be a bool')
                return -1


        #
        # Normalize kernel
        #

        # If not indicated
        if norm is None:

            # Default no
            inorm = False

        # If there is input
        else:

            # Copy input
            inorm = norm

            # Check if bool
            if not isinstance(inorm, bool):
                self.__error('norm must be a bool')
                return -1


        #
        # Convolution kernel
        #

        # If no input
        if kernel is None:

            # Assume Gaussian
            tkernel = 0

        # Specified
        else:

            # Initialize ID
            tkernel = -1

            # Check all valid kernels
            for ii,key in enumerate(self.__vkernelFi):

                # If input in the list
                if kernel.lower() in key:

                    # Identify and leave
                    tkernel = ii
                    break

            # If not found
            if tkernel < 0:
                self.__error('Type of kernel not recognized')
                return -1

            # If Gaussian or square
            if tkernel == 0 or tkernel == 1:

                #
                # Check parameter
                #

                # If not defined
                if par is None:

                    # Error, because it is required
                    self.__error('par is required for kernel ' + \
                                 self.__vkernelFi[tkernel])
                    return -1

                # If there is input
                else:

                    # Copy parameter
                    ipar = par

                    # Check it is float
                    if not isinstance(ipar, float):
                        self.__error('par must be a float number')
                        return -1

                # Copy input axis
                l_PSF = iaxis

            # If custom kernel
            elif tkernel == 2:

                # If Kernel set
                if self.__ck:

                    # Point to kernel data
                    nl = self.__ck_nl
                    l_PSF = self.__ck_l_PSF
                    iPSF = self.__ck_PSF
                    Pmin = np.min(l_PSF)
                    Pmax = np.max(l_PSF)

                # Kernel if not set
                else:

                    #
                    # kernel_file
                    #

                    # If file not specified
                    if kernel_file is None:

                        # Error because it is required
                        self.__error('kernel_file is required ' + \
                                     'for kernel ' + \
                                     self.__vkernelFi[tkernel])
                        return -1

                    # There is input
                    else:

                        # Copy path
                        ikernel_file = kernel_file

                        # Check it is a string
                        if not isinstance(ikernel_file, str):
                            self.__error('kernel_file must be ' + \
                                         'a string')
                            return -1

                    # Try opening the file and handle errors
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

                    # Read kernel data
                    try:
                        bytes = f_kernel.read(4)
                        nl = int(struct.unpack('i', bytes)[0])
                        bytes = f_kernel.read(8*nl)
                        l_PSF = np.array(struct.unpack('d'*nl, bytes), \
                                         dtype=np.float64)
                        bytes = f_kernel.read(8*nl)
                        iPSF = np.array(struct.unpack('d'*nl, bytes), \
                                       dtype=np.float64)
                        Pmin = np.min(l_PSF)
                        Pmax = np.max(l_PSF)
                    except:
                        f_kernel.close()
                        self.__error(f'problem reading kernel in ' + \
                                     f'{kernel_file}')
                        return -1
                    f_kernel.close()

        #
        # Center
        #

        # If no input
        if center is None:

            # If gaussian or square
            if tkernel == 0 or tkernel == 1:

                # Error, because required
                self.__error('center is required for ' + \
                             'kernel ' + self.__vkernelFi[tkernel])
                return -1

        # Specified
        else:

            # Copy
            icenter = center

            # Check it is float
            if not isinstance(icenter, float):
                self.__error('center must be float type')
                return -1


        # Check if equidistant grid
        ds = iaxis[1:] - iaxis[0:-1]

        # If not monotonical
        if np.min(ds) <= 0.:
            self.__error('axis must have only unique values and ' + \
                         'must increase monotonically')
            return -1

        # If the input is not expected equidistant
        if not iequi:

            # Flag as it were
            iequi = True

            # For each distance
            for ii,dsi in enumerate(ds[:-1]):
                # For every other distance
                for dsj in ds[ii+1:]:
                    # If different
                    if np.absolute(dsi - dsj)/(dsi + dsj) > 1e-8:
                        # Flag not equidistant and finish
                        iequi = False
                        break
                # If flagged, we finished
                if not iequi:
                    break


        #
        # Manage axis
        #

        #
        # Get limits

        # Need to extend
        if iextend:

            # Get extrema between both axes
            axis0 = min([np.min(iaxis),np.min(l_PSF)])
            axis1 = max([np.max(iaxis),np.max(l_PSF)])

        # Just input
        else:

            # Get extrema
            axis0 = np.min(iaxis)
            axis1 = np.max(iaxis)

        # If custom, fit to its axis, because no filter outside
        if tkernel == 2:

            # Correct minimum
            if axis0 < Pmin:
                axis0 = Pmin
            # Correct maximum
            if axis1 > Pmax:
                axis1 = Pmax

        # If equidistant
        if iequi:

            # Distance is the same everywhere
            dss = ds[0]

        # Not equidistant
        else:

            # Take minimum
            dss = np.amin(ds)

        # Build equidistant axis between specified limits
        NNi = int((axis1 - axis0)/dss + 1)
        axisi = np.linspace(0,NNi-1,num=NNi, \
                            endpoint=True,dtype=iaxis.dtype)
        dss = (axis1 - axis0)/float(NNi-1)
        axisi *= dss
        axisi += axis0

        #
        # Get filter
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

            PSF = np.interp(axisi, l_PSF, iPSF, left=0.0, \
                            right=0.0)

        # Need to interpolate?
        inter = axis0 != np.min(iaxis) and \
                axis1 != np.max(iaxis) and \
                NNi != iaxis.size

        #
        # Filter
        #

        # If 1D
        if single:

            # Interpolate
            if inter:
                idata = np.interp(axisi, iaxis, idata, \
                                  left=idata[0], right=idata[-1])
            # Filter
            idata *= PSF
            idata = np.trapz(idata,x=axisi)

            # Normalize
            if inorm:
                idata /= np.trapz(PSF,x=axisi)

        # Multi-D
        else:

            # Transform
            idata = np.moveaxis(idata,idim,-1)
            oshape = idata.shape
            idata = idata.reshape(-1,idata.shape[-1])

            # Number of columns
            ncol = idata.shape[0]

            # Interpolate?
            if inter:
                for icol in range(ncol):
                    idata[icol,:] = np.interp(axisi,iaxis, \
                                              idata[icol,:], \
                                              left=idata[icol,0], \
                                              right=idata[icol,-1])
            # Filter
            for icol in range(ncol): idata[icol,:] *= PSF
            idata = np.trapz(idata,x=axisi,axis=1)

            # Normalize
            if inorm:
                idata /= np.trapz(PSF,x=axisi)

            # Transform back
            idata = idata.reshape(oshape[:-1] + (oshape[-1],))
            idata = np.moveaxis(idata,-1,idim)

        return idata

