# Function for 1D, 2D convolutions and 1D filtering

  This code has been used to degrade in the following
  publication:

  del Pino Alem\'an et al. (2018) [2018ApJ...863..164D]

1. degrade.py

   Contains a class with the following methods:

     -convol2D(data, xaxis=None, yaxis=None, xpar=None, \
               ypar=None, xperiod=None, yperiod=None, \
               xextend=None, yextend=None, xdir=None, ydir=None, \
               kernel=None, r0=None, D=None, l0=None, \
               xequi=None, yequi=None)
         * Convolution with a 2D kernel

         o data: 2D numpy array with the data to convolve
         o xaxis: 1D numpy array with the X axis values
         o yaxis: 1D numpy array with the Y axis values
         o xpar: Characteristic parameter of convolution along the
                 X axis. Required for gaussian (fwhm) and square
                 (size of pulse)
         o ypar: Characteristic parameter of convolution along the
                 Y axis. Required for gaussian (fwhm) and square
                 (size of pulse)
         o r0: Characteristic parameter of convolution for seeing
               kernel (seeing parameter)
         o D: Diameter of telescope for airy and seeing kernels
              (D and l0 must have same units)
         o l0: Wavelength of reference for airy and seeing kernels
               (D and l0 must have same units)
         o xperiod: True if the X axis should be assumed to be
                    periodic
         o yperiod: True if the Y axis should be assumed to be
                    periodic
         o xextend: True if the X axis is not periodic and it should
                    be extended to fit the wings of the convolution
                    kernel for the boundary nodes
         o yextend: True if the X axis is not periodic and it should
                    be extended to fit the wings of the convolution
                    kernel for the boundary nodes
         o xdir: Factor that multiplies the characteristic parameter
                 of some convolution kernels (not square) to
                 determine how much the axis must be extended if
                 xextend is True.
         o ydir: Factor that multiplies the characteristic parameter
                 of some convolution kernels (not square) to
                 determine how much the axis must be extended if
                 yextend is True.
         o kernel: Type of convolution kernel. Options are gauss,
                   square, airy and seeing (Fried 1966 long-exposure
                   approximation)
         o xequi: True if the distance between nodes in the X axis
                  is constant. It speeds up the convolution. If the
                  user lies, the result is wrong.
         o yequi: True if the distance between nodes in the Y axis
                  is constant. It speeds up the convolution. If the
                  user lies, the result is wrong.

         * Example: primary mirror degradation.

         kernel = 'seeing'
         r0 = 0.2 # m
         D = 1.5 # m
         # nl number of wavelengths
         # nx number of X axis nodes
         # ny number of Y axis nodes
         # ns number of stokes parameters
         stokes = np.zeros((nl, nx, ny, ns))
         # (...) Fill stokes with data
         lamb = np.zeros((nl))
         # (...) Fill lamb with wavelength axis
         x = np.zeros((nx))
         # (...) Fill x with X axis, equidistant in example
         y = np.zeros((ny))
         # (...) Fill y with Y axis, equidistant in example
         # Get instance of class
         deg = degrade_class()
         # For each wavelength and stokes parameter
         for il,istk in [(il,istk) for il in range(nl) \
                                   for istk in range(ns)]:

             ll = lamb[il]

             # 2D convolution
             stokes[il,:,:,istk] = \
                         deg.convol2D(stokes[il,:,:,istk], \
                                      xaxis=x, yaxis=y, \
                                      xperiod=True, \
                                      yperiod=True, \
                                      xdir=5., ydir=5., \
                                      kernel=kernel, \
                                      r0=r0, D=D, l0=ll)


     -convol1D(data, axis=None, par=None, period=None, extend=None, \
               adir=None, kernel=None, equi=None, kernel_file=None)
         * Convolution with a 1D kernel

         o data: 1D numpy array with the data to convolve
         o axis: 1D numpy array with the abscissa axis values
         o par: Characteristic parameter of convolution. Required for
                gaussian (fwhm) and square (size of pulse)
         o period: True if the abscissa axis should be assumed to be
                    periodic
         o extend: True if the abscissa axis is not periodic and it
                   should be extended to fit the wings of the
                   convolution kernel for the boundary nodes
         o adir: Factor that multiplies the characteristic parameter
                 of some convolution kernels (not square) to
                 determine how much the axis must be extended if
                 extend is True.
         o kernel: Type of convolution kernel. Options are gauss,
                   square, and custom (loads the kernel_file file)
         o equi: True if the distance between nodes in the abscissa
                 axis is constant. It speeds up the convolution. If
                 the user lies, the result is wrong.
         o kernel_file: string with the path to the file with the
                        kernel to be applied. It must be a binary
                        file with the following format:
             $ An integer with the number of data points [nn]
             $ nn double precision numbers with the abscissas
               (same space than the abscissa of the data)
             $ nn double precision numbers with the PSF to
               convolve with.

         * Example: spectral smearing

         # Get instance of class
         deg = degrade_class()
         kernel = 'gauss'
         # fwhm 20 mAngstroms
         par = deg.mAng2m(20)
         # nl number of wavelengths
         # nx number of X axis nodes
         # ny number of Y axis nodes
         # ns number of stokes parameters
         stokes = np.zeros((nl, nx, ny, ns))
         # (...) Fill stokes with data
         lamb = np.zeros((nl))
         # (...) Fill lamb with wavelength axis
         x = np.zeros((nx))
         # (...) Fill x with X axis, equidistant in example
         y = np.zeros((ny))
         # (...) Fill y with Y axis, equidistant in example
         # It is recommended to refine the wavelength axis
         # for a step of, for example, 1 mAngstrom
         dl = deg.mAng2m(1.0)
         nl1 = int((np.amax(lamb) - np.amin(lamb))/dl + 1)
         lamb1 = np.linspace(0,nl1-1,num=nl1,dtype=lamb.dtype)
         lamb1 *= dl
         lamb1 += np.amin(lamb)
         equi = True
         # For each spatial node and stokes parameter
         for iy,ix,istk in [(iy,ix,istk) for iy in range(ny) \
                                         for ix in range(nx) \
                                         for istk in range(ns)]:

             # Get the vector to degrade
             stkl = np.copy(stokes[:,iy,ix,istk])

             # Refine
             stkl = np.interp(lamb1,lamb,stkl)

             # Convolution
             stkl = deg.convol1D(stkl, axis=lamb1, \
                                 par=par, extend=True, \
                                 adir=5., kernel=kernel, \
                                 equi=equi)

             # Coarse
             stkl = np.interp(lamb,lamb1,stkl) 

             # Store in array
             stokes[:,iy,ix,istk] = stkl

     -filter1D(data, axis=None, par=None, center=None, extend=None, \
               kernel=None, equi=None, kernel_file=None, norm=None)
         * Integral weighted with a 1D filter

         o data: 1D numpy array with the data to convolve
         o axis: 1D numpy array with the abscissa axis values
         o par: Characteristic parameter of convolution. Required for
                gaussian (fwhm) and square (size of pulse)
         o center: Center of the filter in the abscissa axis. Required
                   for gauss and square
         o extend: True if the abscissa axis is not periodic and it
                   should be extended to fit the wings of the
                   convolution kernel for the boundary nodes
         o kernel: Type of convolution kernel. Options are gauss,
                   square, and custom (loads the kernel_file file)
         o equi: True if the distance between nodes in the abscissa
                 axis is constant. It speeds up the convolution. If
                 the user lies, the result is wrong.
         o kernel_file: string with the path to the file with the
                        kernel to be applied. It must be a binary
                        file with the following format:
             $ An integer with the number of data points [nn]
             $ nn double precision numbers with the abscissas
               (same space than the abscissa of the data)
             $ nn double precision numbers with the PSF to
               convolve with.
         o norm: If True, normalizes the integral to the integral of
                 the filter

         * Example: spectral filtergram custom filter

         # Get instance of class
         deg = degrade_class()
         kernel = 'custom'
         kernel_file = 'path_to_file'
         # nl number of wavelengths
         # nx number of X axis nodes
         # ny number of Y axis nodes
         # ns number of stokes parameters
         stokes = np.zeros((nl, nx, ny, ns))
         # (...) Fill stokes with data
         lamb = np.zeros((nl))
         # (...) Fill lamb with wavelength axis
         x = np.zeros((nx))
         # (...) Fill x with X axis, equidistant in example
         y = np.zeros((ny))
         # (...) Fill y with Y axis, equidistant in example
         # It is recommended to refine the wavelength axis
         # for a step of, for example, 1 mAngstrom
         dl = deg.mAng2m(1.0)
         nl1 = int((np.amax(lamb) - np.amin(lamb))/dl + 1)
         lamb1 = np.linspace(0,nl1-1,num=nl1,dtype=lamb.dtype)
         lamb1 *= dl
         lamb1 += np.amin(lamb)
         equi = True
         # For each spatial node and stokes parameter
         for iy,ix,istk in [(iy,ix,istk) for iy in range(ny) \
                                         for ix in range(nx) \
                                         for istk in range(ns)]:

             # Get the vector to degrade
             stkl = np.copy(stokes[:,iy,ix,istk])

             # Refine
             stkl = np.interp(lamb1,lamb,stkl)

             # Filter
             stkl = deg.filter1D(stkl, axis=lamb1, \
                                 extend=True, \
                                 kernel=kernel, equi=equi, \
                                 kernel_file = kernel_file)

             # Coarse
             stkl = np.interp(lamb,lamb1,stkl) 

             # Store in array
             stokes[0,iy,ix,istk] = stkl

         # Remove integrated dimension
         stokes = stokes[0,:,:,:]

     - cm2sec(x): Takes x in cm on the solar surface and returns the
                  corresponding arcseconds.
     - m2sec(x): Takes x in m on the solar surface and returns the
                 corresponding arcseconds.
     - km2sec(x): Takes x in km on the solar surface and returns the
                  corresponding arcseconds.
     - Mm2sec(x): Takes x in Mm on the solar surface and returns the
                  corresponding arcseconds
     - sec2Mm(x): Takes x in arcseconds and returns the corresponding
                  Mm on the solar surface.
     - fwhm2sig(x): Takes x in full width half maximum and returns the
                    corresponding sigma parameter.
     - deg2ra(x): Takes x in degrees and returns the corresponding
                  radians.
     - sec2ra(x): Takes x in seconds and returns the corresponding
                  radians.
     - ra2deg(x): Takes x in radians and returns the corresponding
                  degrees.
     - ra2sec(x): Takes x in radians and returns the corresponding
                  seconds.
     - nm2m(x): Takes x in nm and returns the corresponding m.
     - Ang2m(x): Takes x in angstroms and returns the corresponding m.
     - m2Ang(x): Takes x in m and returns the corresponding angstroms.
     - mAng2m(x): Takes x in miliangstroms and returns the
                  corresponding m.

# License

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 
###
 
 # TO DO

  - Better readme, probably.
