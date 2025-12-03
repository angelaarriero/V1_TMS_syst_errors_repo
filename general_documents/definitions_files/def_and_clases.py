from sympy import *
init_printing()
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
from sympy import re, im, I, E, symbols
######################################################################################################################
####### Here I define the symbolic variables with sympy
k1,k2,p=symbols('k1,k2,p')
g1,g2,g3,g4,xsky,xload,ysky,yload,g=symbols('g1,g2,g3,g4,xsky,xload,ysky,yload,g')
n1,n2,n3,n4,n=symbols('n1,n2,n3,n4,n')
g1c,g2c,g3c,g4c,xskyc,xloadc,yskyc,yloadc=symbols('g1c,g2c,g3c,g4c,xskyc,xloadc,yskyc,yloadc')
Tn1,Tn2,Tn3,Tn4=symbols('Tn1,Tn2,Tn3,Tn4')
g1Fc,g2Fc,g3Fc,g4Fc=symbols('g1Fc,g2Fc,g3Fc,g4Fc')
k,v1,v2,v3,v4=symbols('k,v1,v2,v3,v4')
#b1,b2,b3,b4=symbols('b1,b2,b3,b4')
B1,TnF1,TnF2,TnF3,TnF4=symbols('B1,TnF1,TnF2,TnF3,TnF4')
Isky,Qsky,Usky,Vsky=symbols('Isky,Qsky,Usky,Vsky')
Iload,Qload,Uload,Vload=symbols('Iload,Qload,Uload,Vload')
theta_os1,theta_os2,theta_os3=symbols('theta_os1,theta_os2,theta_os3', real=True)
theta_ol1,theta_ol2,theta_ol3=symbols('theta_ol1,theta_ol2,theta_ol3', real=True)
beta_h11_1,beta_h11_2,beta_h11_3=symbols('beta_h11_1,beta_h11_2,beta_h11_3', real=True)
beta_h12_1,beta_h12_2,beta_h12_3=symbols('beta_h12_1,beta_h12_2,beta_h12_3', real=True)
theta_lna3_4,theta_lna1_2=symbols('theta_lna3_4,theta_lna1_2', real=True)
Osx,Osy,Osa,Olx,Oly,Ola=symbols('Osx,Osy,Osa,Olx,Oly,Ola')
### omt sky
lmd_os1,lmd_os2,e_os,phi_os1,phi_os2,phi_os3=symbols('lambda_os1,lambda_os2,epsilon_os,phi_os1,phi_os2,phi_os3')
phi_os1c,phi_os2c,phi_os3c=symbols('phi_os1c,phi_os2c,phi_os3c')
### omt load
lmd_ol1,lmd_ol2,e_ol,phi_ol1,phi_ol2,phi_ol3=symbols('lambda_ol1,lambda_ol2,epsilon_ol,phi_ol1,phi_ol2,phi_ol3')
phi_ol1c,phi_ol2c,phi_ol3c=symbols('phi_ol1c,phi_ol2c,phi_ol3c')
### HYB 11
lmd_h11_1,lmd_h11_4,e_h11,phi_h11_1,phi_h11_2,phi_h11_3=symbols('lambda_h11_1,lambda_h11_4,e_h11,phi_h11_1,phi_h11_2,phi_h11_3')
phi_h11_1c,phi_h11_2c,phi_h11_3c,phi_h11_4c=symbols('phi_h11_1c,phi_h11_2c,phi_h11_3c,phi_h11_4c')
### HYB 12
lmd_h12_1,lmd_h12_4,e_h12,phi_h12_1,phi_h12_2,phi_h12_3=symbols('lambda_h12_1,lambda_h12_4,e_h12,phi_h12_1,phi_h12_2,phi_h12_3')
phi_h12_1c,phi_h12_2c,phi_h12_3c,phi_h12_4c=symbols('phi_h12_1c,phi_h12_2c,phi_h12_3c,phi_h12_4c')
##### lna
spi_lna1_2,spi_lna3_4,phi_lna1_2,phi_lna3_4=symbols('spi_lna1_2,spi_lna3_4,phi_lna1_2,phi_lna3_4')
phi_lna1_2c,phi_lna3_4c=symbols('phi_lna1_2c,phi_lna3_4c')
###### BEM
g1F,g2F,g3F,g4F,n1F,n2F,n3F,n4F,Tn1F,Tn2F,Tn3F,Tn4F=symbols('g1F,g2F,g3F,g4F,n1F,n2F,n3F,n4F,Tn1F,Tn2F,Tn3F,Tn4F')
### LO+MIXER
g1LO,g2LO,g3LO,g4LO,n1M,n2M,n3M,n4M,Tn1M,Tn2M,Tn3M,Tn4M=symbols('g1LO,g2LO,g3LO,g4LO,n1M,n2M,n3M,n4M,Tn1M,Tn2M,Tn3M,Tn4M')
g1M,g2M,g3M,g4M=symbols('g1M,g2M,g3M,g4M')
#### DOWN-CONVERTER
g1DC,g2DC,g3DC,g4DC,n1DC,n2DC,n3DC,n4DC,Tn1DC,Tn2DC,Tn3DC,Tn4DC=symbols('g1DC,g2DC,g3DC,g4DC,n1DC,n2DC,n3DC,n4DC,Tn1DC,Tn2DC,Tn3DC,Tn4DC')
#########################################################################################################################
##############################################################################################################################


######### c means a complex number: |x|²
def modulosCuadrados(Z,Zc):
    Total=Z*Zc
    return Total
######### here I replace the electric field with the stokes parameter that corresponds
def reemplazo_stokes(a):
    aexp=expand(a)
    b=aexp.subs([(xsky*xskyc,'(Isky+Qsky)/2'),(ysky*yskyc,'(Isky-Qsky)/2'),(xload*xloadc,'(Iload+Qload)/2'),
   (yload*yloadc,'(Iload-Qload)/2'),(xsky*yskyc,'(Usky+I*Vsky)/2'),(xskyc*ysky,'(Usky-I*Vsky)/2'),
    (xload*yloadc,'(Uload+I*Vload)/2'),(xloadc*yload,'(Uload-I*Vload)/2'),(xload*xskyc,'0'),
     (yload*yskyc,'0'),(xloadc*xsky,'0'),(yloadc*ysky,'0'),(xload*yskyc,'0'),(xsky*yloadc,'0'),(xskyc*yload,'0'),(xloadc*ysky,'0'),
               (n1**2,'Tn1'),(n2**2,'Tn2'),(n3**2,'Tn3'),(n4**2,'Tn4'),(n1*n2,'0'),
                (n2*n3,'0'),(n3*n4,'0'),(n4*n1,'0'),(n1*n3,'0'),(n4*n2,'0'),(n1,'0'),(n2,'0'),(n3,'0'),(n4,'0'),
              (n1F**2,'TnF1'),(n2F**2,'TnF2'),(n3F**2,'TnF3'),(n4F**2,'TnF4'),(n1F*n2F,'0'),
                (n2F*n3F,'0'),(n3F*n4F,'0'),(n4F*n1F,'0'),(n1F*n3F,'0'),(n4F*n2F,'0'),(n1F,'0'),(n2F,'0'),(n3F,'0'),(n4F,'0'),
              (n1M**2,'Tn1M'),(n2M**2,'Tn2M'),(n3M**2,'Tn3M'),(n4M**2,'Tn4M'),(n1M*n2M,'0'),
                (n2M*n3M,'0'),(n3M*n4M,'0'),(n4M*n1M,'0'),(n1M*n3M,'0'),(n4M*n2M,'0'),(n1M,'0'),(n2M,'0'),(n3M,'0'),(n4M,'0'),
              (n1DC**2,'Tn1DC'),(n2DC**2,'Tn2DC'),(n3DC**2,'Tn3DC'),(n4DC**2,'Tn4DC'),(n1DC*n2DC,'0'),
                (n2DC*n3DC,'0'),(n3DC*n4DC,'0'),(n4DC*n1DC,'0'),(n1DC*n3DC,'0'),(n4DC*n2DC,'0'),
               (n1DC,'0'),(n2DC,'0'),(n3DC,'0'),(n4DC,'0')])
    return b
########### define the conjugate of the real value
def reemplazo_conjugados(a):
    b=a.subs([(ysky,'yskyc'),(xsky,'xskyc'),(yload,'yloadc'),(xload,'xloadc'),(phi_os1,'phi_os1c'),
   (phi_os2,'phi_os2c'),(phi_os3,'phi_os3c'),(phi_ol1,'phi_ol1c'),(phi_ol2,'phi_ol2c'),(phi_ol3,'phi_ol3c'),
            (phi_h11_1,'phi_h11_1c'),(phi_h11_2,'phi_h11_2c'),(phi_h11_3,'phi_h11_3c'),
            (phi_h12_2,'phi_h12_2c'),(phi_h12_3,'phi_h12_3c'),(phi_lna1_2,'phi_lna1_2c'),(phi_lna3_4,'phi_lna3_4c'),
            (spi_lna1_2,'spi_lna1_2c'),(spi_lna3_4,'spi_lna3_4c')])
    return b
####### help to replace the non-idealities with 0 or 1 to check the final output.
def reemplazo_unidad(a):
    b=a.subs([(lmd_os1,'0'),(lmd_os2,'0'),(e_os,'0'),(phi_os1,'1'),(phi_os2,'1'),
   (phi_os3,'1'),(lmd_ol1,'0'),(lmd_ol2,'0'),(e_ol,'0'),(phi_ol1,'1'),(phi_ol2,'1'),
    (phi_ol3,'1'),(lmd_h11_1,'0'),(lmd_h11_4,'0'),(e_h11,'0'),
         (phi_h11_1,'1'),(phi_h11_2,'1'),(phi_h11_3,'1'),
            (lmd_h12_1,'0'),(lmd_h12_4,'0'),(e_h12,'0'),(phi_h12_1,'1'),
             (phi_h12_2,'1'),(phi_h12_3,'1'),
             (spi_lna1_2,'0'),(spi_lna3_4,'0'),(phi_lna1_2,'1'),(phi_lna3_4,'1'),
            (n1*n2,'0'),(n2*n3,'0'),(n3*n4,'0'),(n4*n1,'0'),(n1*n3,'0'),(n4*n2,'0'),
            (n1,'0'),(n2,'0'),(n3,'0'),(n4,'0'),(g1,'1'),(g2,'1'),(g3,'1'),(g4,'1'),
            (Tn1,'0'),(Tn2,'0'),(Tn3,'0'),(Tn4,'0'),(theta_os1,'0'),(theta_os2,'0'),(theta_os3,'0'),
           (theta_ol1,'0'),(theta_ol2,'0'),(theta_ol3,'0'),(k1,'1')])
    return b

def reemplazo_names(a):
    b=a.subs([(lmd_os1,'Osx'),(lmd_os2,'Osy'),(e_os,'Osa'),(lmd_ol1,'Olx'),(lmd_ol2,'Oly'),(e_ol,'Ola'),
             (lmd_h11_1,'B11'),(lmd_h11_4,'B14'),(e_h11,'B1a'),(lmd_h12_1,'B21'),(lmd_h12_4,'B24'),(e_h12,'B2a')])
    return b

####### Here we change e^(ix) in terms of cos(x) and sin(x) 
def reemplazo_fases(output):
    phi_os2_re=cos(theta_os2)+1j*sin(theta_os2);phi_os3_re=cos(theta_os3)+1j*sin(theta_os3);phi_os1_re=phi_os2_re*phi_os3_re;
    phi_ol2_re=cos(theta_ol2)+1j*sin(theta_ol2);phi_ol3_re=cos(theta_ol3)+1j*sin(theta_ol3);phi_ol1_re=phi_ol2_re*phi_ol3_re;
    a1=phi_os1_re.conjugate();a2=phi_os2_re.conjugate();a3=phi_os3_re.conjugate();
    a4=phi_ol1_re.conjugate();a5=phi_ol2_re.conjugate();a6=phi_ol3_re.conjugate();
    phi_h11_1_re=cos(beta_h11_1)+1j*sin(beta_h11_1);phi_h11_2_re=cos(beta_h11_2)+1j*sin(beta_h11_2);
    phi_h11_3_re=phi_h11_1_re*phi_h11_2_re;
    phi_h12_1_re=cos(beta_h12_1)+1j*sin(beta_h12_1);phi_h12_2_re=cos(beta_h12_2)+1j*sin(beta_h12_2);
    phi_h12_3_re=phi_h12_1_re*phi_h12_2_re; a7=phi_h11_1_re.conjugate();a8=phi_h11_2_re.conjugate();
    a9=phi_h11_3_re.conjugate();a11=phi_h12_1_re.conjugate();a12=phi_h12_2_re.conjugate();a13=phi_h12_3_re.conjugate();
    phi_lna1_2_re=cos(theta_lna1_2)+1j*sin(theta_lna1_2);phi_lna3_4_re=cos(theta_lna3_4)+1j*sin(theta_lna3_4);
    a15=phi_lna1_2_re.conjugate();a16=phi_lna3_4_re.conjugate();
    f=output.subs([(phi_os1,phi_os1_re),(phi_os1c,a1),(phi_os2,phi_os2_re),(phi_os2c,a2),
                   (phi_os3,phi_os3_re),(phi_os3c,a3),(phi_ol1,phi_ol1_re),(phi_ol1c,a4),
                   (phi_ol2,phi_ol2_re),(phi_ol2c,a5),(phi_ol3,phi_ol3_re),(phi_ol3c,a6),
                   (phi_h11_1,phi_h11_1_re),(phi_h11_1c,a7),(phi_h11_2,phi_h11_2_re),(phi_h11_2c,a8),
                   (phi_h11_3,phi_h11_3_re),(phi_h11_3c,a9),(phi_h12_1,phi_h12_1_re),(phi_h12_1c,a11),
                   (phi_h12_2,phi_h12_2_re), (phi_h12_2c,a12),(phi_h12_3,phi_h12_3_re),(phi_h12_3c,a13),
                   (phi_lna1_2,phi_lna1_2_re),(phi_lna1_2c,a15),(phi_lna3_4,phi_lna3_4_re),(phi_lna3_4c,a16)])
    return f

#########################################################################################################################
############################ JONES MATRICES: here we code the path of the signal through the system with matrices         #######################################################################################################################
#######
def RF_components_behavior(signal_W,signal_IRf,OMTsky,OMTload,HYB11,HYB12,LNA11,LNA12,n1_2,n3_4,att_signal1,att_signal2,LNA11B,LNA12B,n1_2B,n3_4B):
    #### optical components
    wind_at1=np.dot(signal_W,att_signal1) # WINDOW
    IRf_at2=np.dot(wind_at1,att_signal2) # IR FILTER
    #print('IRf_at2',IRf_at2)
    a0_1=np.dot(IRf_at2,OMTsky)         # 
    #### omt
    sky_s=np.array([[xsky],[ysky]])
    load_s=np.array([[xload],[yload]])
    
    a1_2=np.dot(a0_1,sky_s)
    a3_4=np.dot(OMTload,load_s)

    a1o=a1_2[0]
    a2o=a1_2[1]
    a3o=a3_4[0]
    a4o=a3_4[1]

    #### hyb
    #print('a1o',a1o)
    #print('a3o',a3o)
    #print('[a1o,a3o]',np.array([a1o,a3o]))
    b1_2=np.dot(HYB11,np.array([a1o,a3o]))
    b3_4=np.dot(HYB12,np.array([a2o,a4o]))
    #print('b1_2',b1_2)
    #print('b3_4',b3_4)

    ## LNA
    A1_2=np.dot(LNA11,(b1_2+n1_2))
    B1_2=np.dot(LNA12,(b3_4+n3_4))
    
    ### BEM
    A1_2_B=np.dot(LNA11B,(A1_2+n1_2B))
    B1_2_B=np.dot(LNA12B,(B1_2+n3_4B))
    ###
    
    
    ######### digital HYB2 #######
    HYB2=(1/np.sqrt(2))*np.array([[1,1],[1,-1]])
    C1_2=np.dot(HYB2,A1_2_B)
    D1_2=np.dot(HYB2,B1_2_B)

    C1_tot=C1_2[0]
    C2_tot=C1_2[1]
    D1_tot=D1_2[0]
    D2_tot=D1_2[1]

    return C1_tot,C2_tot,D1_tot,D2_tot,A1_2,B1_2
#########################################################################################################################
############################ In this part, the signal is detected and setting it into the stokes parameters.                          
#########################################################################################################################

def TMS_results(C1_tot,C2_tot,D1_tot,D2_tot):
  ### para las ecuaciones numeros complejos
  C1Uc=reemplazo_conjugados(sp.Matrix(C1_tot))
  C2Uc=reemplazo_conjugados(sp.Matrix(C2_tot))
  D1Uc=reemplazo_conjugados(sp.Matrix(D1_tot))
  D2Uc=reemplazo_conjugados(sp.Matrix(D2_tot))

  ### reemplazo fases
  C1_tot2=reemplazo_fases(sp.Matrix(C1_tot))
  C1Uc2=reemplazo_fases(C1Uc)
  C2_tot2=reemplazo_fases(sp.Matrix(C2_tot))
  C2Uc2=reemplazo_fases(C2Uc)
  D1_tot2=reemplazo_fases(sp.Matrix(D1_tot))
  D1Uc2=reemplazo_fases(D1Uc)
  D2_tot2=reemplazo_fases(sp.Matrix(D2_tot))
  D2Uc2=reemplazo_fases(D2Uc)


  #### modulo cuadrado de C1, C2, D1, D2
  totalC1=modulosCuadrados(sp.Matrix(C1_tot2),C1Uc2)
  totalC2=modulosCuadrados(sp.Matrix(C2_tot2),C2Uc2)
  totalD1=modulosCuadrados(sp.Matrix(D1_tot2),D1Uc2)
  totalD2=modulosCuadrados(sp.Matrix(D2_tot2),D2Uc2)
  #print('totalC1',totalC1)
  #print('totalC2',totalC2)
  #print('totalD1',totalD1)
  #print('totalD2',totalD2)

  #### Obtener parametros de stokes luego del modulo cuadrado
  C1totalPower=reemplazo_stokes(totalC1)
  C2totalPower=reemplazo_stokes(totalC2)
  D1totalPower=reemplazo_stokes(totalD1)
  D2totalPower=reemplazo_stokes(totalD2)
  E1U=sp.Matrix(C1_tot2)*D1Uc2
  E2U=sp.Matrix(D1_tot2)*C1Uc2
  F1U=sp.Matrix(C2_tot2)*D2Uc2
  F2U=sp.Matrix(D2_tot2)*C2Uc2


  #### SALIDAS DE LA FPGA O1 .. O8
  O1=C1totalPower+D1totalPower
  O2=C1totalPower-D1totalPower
  O3=reemplazo_stokes(E1U+E2U)
  O4=reemplazo_stokes(E1U-E2U)*(-I)
  O5=C2totalPower+D2totalPower
  O6=C2totalPower-D2totalPower
  O7=reemplazo_stokes(F1U+F2U)
  O8=reemplazo_stokes(F1U-F2U)*(-I)

  return O1,O2,O3,O4,O5,O6,O7,O8