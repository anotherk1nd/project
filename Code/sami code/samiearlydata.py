from astropy.io import fits
import scipy as sp
from astropy.io.fits import getheader

fn = 'C:\Users\Joshua\Documents\Term 1\Project\Data\Sami\earlyrelease.fits'
hdulist = fits.open(fn)
hdulist.info()
colnames = hdulist[0]
print colnames
tbdata = hdulist[1]
print tbdata
cols = hdulist[1].columns
cols.info()
cols.names
getheader(fn)  # get default HDU (=0), i.e. primary HDU's header
getheader(fn, 0)  # get primary HDU's header
getheader(fn, 1)  # the second extension

colinfo = hdulist[0].data # gives column names, info
print colinfo
data = hdulist[1].data
#newdata =sp.zeros(108,24)
stuff = data.field(0) # CONTAINS ALL INFO AS 1D ARRAY
#print data.field(0)[1:,:]
"""for i in range():
    newdata = float(line.strip())
    print data 
"""
print stuff
print stuff[1]
print type(stuff[1]) # This is a str
print type(stuff) # This is an array - it is an array with strings for each entry
stuff2 = stuff.split() # this takes the long str and turns it into a list
print 'WOOOOOOOO'
print stuff2
print len(stuff2) # 108 entries long, 1 for each galaxy
print type(stuff2) # This is an array 
print type(stuff2[1]) # This is a list - WE NEED TO CONVERT THIS LIST INTO AN ARRAY ALSO SO THAT WE HAVE ARRAYS OF ARRAYS, TO DO SO WE CONVERT BACK TO A STRING AND USE SAME PROCESS

str1 = ' '.join(stuff2[1]) #This is a str of 1 entry that contains all the data
print 'str1',str1
print type(str1) # This is a str
str1split = str1.split()
print str1split
print type(str1split) # This is a list again
print len(str1split) # This contains all the entries as separate entries of a list
print str1split[0]
#print stuff2
print 'woooop'
#print newpoints[0]
newer = sp.zeros([107,18])
newer[0,0] = str1split[1]
newer[0,1] = str1split[2]

for j in range(1,108):
    str1 = ' '.join(stuff2[j])
    str1split = str1.split()
    for i in range(0,18):
        print str1split[i+1]
        newer[j-1,i] = str1split[i+1]
print newer
print newer[:,16]

fnlam = 'C:\Users\Joshua\Documents\Term 1\Project\Data\Sami\lambdar_SAMI_LC2016.fits' # This contains the lambda values the dude sent me
hdulist = fits.open(fnlam)
hdulist.info()
colnames = hdulist[0]
print colnames
tbdata = hdulist[1]
print tbdata
cols = hdulist[1].columns
cols.info()
cols.names
getheader(fnlam)  # get default HDU (=0), i.e. primary HDU's header
getheader(fnlam, 0)  # get primary HDU's header
getheader(fnlam, 1)  # the second extension

colinfo = hdulist[0].data # gives column names, info
print colinfo
data = hdulist[1].data
#newdata =sp.zeros(108,24)
print data.field(0) # CONTAINS ALL INFO AS 1D ARRAY
print len(data.field(0))
print newer[:,16]
print len(newer[:,16])


"""
print newer
for j in range(106):
    for i in range(1,17): # PROBLEM HERE, I START AT 1 TO AVOID THE NAME IN THE ARRAY BUT THAT MESSES UP OTHER STUFF
        str1 = ' '.join(stuff2[i])
        print str1
        str1split = str1.split()
        print type(str1split)
        print str1split
        print str1split[i]
        newer[j,i-1] = str1split[i]
        print 'I=',i
        print 'j=',j
        print newer
#newer = stuff2[1].split()
print newer[0,3]
"""
