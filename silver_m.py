#!/usr/bin/python

#Math
import pylab
from   numpy import sqrt


#Custom
import SFP
import SFP_dielectrics 


#----------------------------
# CODE BEGINS
# ---------------------------

function=SFP_dielectrics.Ag_JC_SS

m=1.333
MFP=500.0
loss=SFP_dielectrics.Ag_path_to_g(MFP)
filename="Ag_MFP_%3d_m%4d.tab"%(MFP,m*1000)

minwave=200.0
maxwave=1600.0
dwave  =5.0

print "MFP  = ",MFP
print "Loss = ",loss


SFP_dielectrics.generate_tab(filename,minwave,maxwave,dwave,function,m=m,gscat=loss)

