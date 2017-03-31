import astropy
#astropy.test
from astropy.io import fits
import scipy
gxylist = fits.open('atlas3d.fits')
gxylist.info()