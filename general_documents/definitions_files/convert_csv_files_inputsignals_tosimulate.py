import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv
from numpy.fft import fft

def datos_componente(nombre_archivo,n):
    data = []
    with open(nombre_archivo, 'r') as file: 
        csv_reader = csv.reader(file, quotechar='"')
        for row in csv_reader:
            aux = []
            for i in range(0, len(row)):
                aux.append(float(row[i]))
            data.append(aux)
    file.close()
    
    freq=[]
    data_res=[]
    for i in range(len(data)):
        freq.append(data[i][0])
        data_res.append(data[i][1])
        
    xdup_freq = freq
    ydup_dat = data_res
    i_dat=[]
    for i in xdup_freq:
        while(xdup_freq.count(i) > 1):
            xdup_freq.remove(i)
            i_dat.append(xdup_freq.index(i))
    for i in range(len(i_dat)):
        ydup_dat.remove(ydup_dat[i_dat[i]]) 
        
    x = xdup_freq
    y = ydup_dat
    f1 = interp1d(x, y, kind='nearest')
    xx = np.linspace(10,20,n)#np.linspace(min(x),max(x),n)
    #xx = np.linspace(min(x),max(x),n)
    
    return freq, data_res,xdup_freq,ydup_dat,xx,f1(xx)

def datos_componente2(nombre_archivo,n):
    data = []
    with open(nombre_archivo, 'r') as file: 
        csv_reader = csv.reader(file, quotechar='"')
        for row in csv_reader:
            aux = []
            for i in range(0, len(row)):
                aux.append(float(row[i]))
            data.append(aux)
    file.close()
    
    freq=[]
    data_res=[]
    for i in range(len(data)):
        freq.append(data[i][0])
        data_res.append(data[i][1])
        
    xdup_freq = freq
    ydup_dat = data_res
    i_dat=[]
    for i in xdup_freq:
        while(xdup_freq.count(i) > 1):
            xdup_freq.remove(i)
            i_dat.append(xdup_freq.index(i))
    for i in range(len(i_dat)):
        ydup_dat.remove(ydup_dat[i_dat[i]]) 
        
    x = xdup_freq
    y = ydup_dat
    f1 = interp1d(x, y, kind='nearest')
    #xx = np.linspace(10,20,n)#np.linspace(min(x),max(x),n)
    xx = np.linspace(min(x),max(x),n)
    
    return freq, data_res,xdup_freq,ydup_dat,xx,f1(xx)

def sparams_to_power(values,n):
    transf_val=[]
    for i in range(n):
        transf_val.append(pow(10,(values[i]/10)))
    return transf_val

def datos_simulados_RI(n,rl_hyb,IL_hyb,IL_omt,RL_omt,RL_feedhorn,RL_window,gain_LNA,RL_load,Tn_lna,noise_lna_amp_roomT):#,noise_lna_amp_roomT
    
    ## DATOS TOMADOS DE LAS SIMULACIONES EN CST DE LA TESIS DE PAZ, PARA OMT, HIBRIDO Y FEEDHORN
    ### n SIGNIFICA QUE TENDREMOS 200 DATOS DESDE 10 A 20 GHz
    # DATOS DEL HYB
    freq1, data_res1,xdup_freq1,ydup_dat1,xx1,data_RL_hyb=datos_componente(rl_hyb,n) #rl
    freq2, data_res2,xdup_freq2,ydup_dat2,xx2,data_IL_hyb=datos_componente(IL_hyb,n) #il

    # DATOS DEL OMT
    freq3, data_res3,xdup_freq3,ydup_dat3,xx3,data_IL_omt=datos_componente(IL_omt,n) #il
    freq4, data_res4,xdup_freq4,ydup_dat4,xx4,data_RL_omt=datos_componente(RL_omt,n) #rl

    # DATOS DEL FEEDHORN
    freq5, data_res5,xdup_freq5,ydup_dat5,xx5,data_RL_FH =datos_componente(RL_feedhorn,n) #rl
    #dtos de la ventana
    freq6, data_res6,xdup_freq6,ydup_dat6,xx6,data_RL_W =datos_componente(RL_window,n) #rl
    #dtos de gain lna
    freq7, data_res7,xdup_freq7,ydup_dat7,xx7,data_gain_lna =datos_componente(gain_LNA,n) #rl
     #dtos de load
    freq8, data_res8,xdup_freq8,ydup_dat8,xx8,data_RL_load =datos_componente(RL_load,n) #rl
     #dtos de tnoise lna
    freq9, data_res9,xdup_freq9,ydup_dat9,xx9,data_tnoise_lna =datos_componente(Tn_lna,n) #TNLNA
    
    #dtos de tnoise lna
    freq10, data_res10,xdup_freq10,ydup_dat10,xx10,noise_lna_amp_roomT =datos_componente(noise_lna_amp_roomT,n) #TNLNAROOM
    

    #conversion de los datos de la tesis de paz en db a potencia [W]
    new_rl_hyb_real=sparams_to_power(data_RL_hyb,n)
    new_il_hyb_real=sparams_to_power(data_IL_hyb,n)
    new_rl_omt_real=sparams_to_power(data_RL_omt,n)
    new_il_omt_real=sparams_to_power(data_IL_omt,n)
    new_rl_fh_real=sparams_to_power(data_RL_FH,n)
    new_rl_w_real=sparams_to_power(data_RL_W,n)
    new_gain_lna_real=sparams_to_power(data_gain_lna,n)
    new_rl_load_real=sparams_to_power(data_RL_load,n)

    ## para simular que pasa si tenemos valores de IL OMT E IL HYB divididos entre 2
    Q=[]
    W=[]
    E=[]
    R=[]
    
    Z=[]
    V=[]
    U1=[]
    U2=[]
    U3=[]
    U4=[]
    N1=[]
    N2=[]
    N3=[]
    N4=[]
    
    sfN1=[]
    
    C=[]
    for i in range(n):
        Q.append(new_il_omt_real[i]/5)# 0.1
        W.append(new_il_omt_real[i]/10)# 0.05
        E.append((new_il_omt_real[i]/150)+0.0002)##0.003
        R.append(new_il_omt_real[i]/42)##0.01  50
        #C.append(new_rl_hyb_real[i]/10)##0.01
        #Z.append(new_rl_w_real[i]/100)##0.01
        #V.append(new_rl_fh_real[i]/100)##0.01
        U1.append(data_tnoise_lna[i]+1e-3)##0.01 #### NOISE LNA
        U2.append(data_tnoise_lna[i]-1e-3)##0.01
        U3.append(data_tnoise_lna[i]+0.9e-3)##0.01
        U4.append(data_tnoise_lna[i]-0.8e-3)##0.01
        #############################################
        N1.append(new_gain_lna_real[i]/0.81)## #### GAIN LNA
        N2.append(new_gain_lna_real[i]/0.80)##
        N3.append(new_gain_lna_real[i]/1.26)##
        N4.append(new_gain_lna_real[i]/1.27)##
        sfN1.append(new_gain_lna_real[i]*np.sin(0.4))##
        
    Q2=[]
    W2=[]
    E2=[]
    R2=[]
    for i in range(n):
        Q2.append(1-pow(10,(-Q[i]/10)))# 
        W2.append(1-pow(10,(-W[i]/10)))#
        E2.append(1-pow(10,(-E[i]/10)))#
        R2.append(1-pow(10,(-R[i]/10)))#
    Z=new_rl_w_real ### WINDOW
    X=new_rl_omt_real ### OMT
    C=new_rl_hyb_real ### HYB
    V=new_rl_fh_real ### FEEDHORN
    M=new_rl_load_real ### LOAD
    N=new_gain_lna_real
    U=data_tnoise_lna
    
    return Q2,W2,E2,R2,Z,X,C,V,xx1,data_RL_W,data_RL_FH,data_RL_omt,data_IL_omt,data_IL_hyb,data_RL_hyb,data_gain_lna, N,M,data_RL_load,U,Q,W,E,R,noise_lna_amp_roomT,N1,N2,N3,N4,U1,U2,U3,U4,sfN1###,noise_lna_amp_roomT


def conversion_dc(f13_file,n):
    freq13, data_res13,xdup_freq13,ydup_dat13,xx13,Noise_amp_DC =datos_componente2(f13_file,n) #NOISE ampl DC
    new_Noise_DC_troom=sparams_to_power(Noise_amp_DC,n) ###
    
    noise_DC_cal=[]
    for i in range(n):
        noise_DC_cal.append(((pow(10,(new_Noise_DC_troom[i])/10)-1)*300))
        
    return noise_DC_cal

def suavizar_picos(y, umbral_factor=2):
    """
    Reemplaza puntos absurdamente altos comparados con vecinos inmediatos.
    
    umbral_factor: cuántas veces mayor debe ser para ser considerado "absurdo"
    """
    y = y.copy()
    for i in range(1, len(y)-1):
        vec = 0.5*(y[i-1] + y[i+1])
        if np.abs(y[i]) > umbral_factor * np.abs(vec):
            y[i] = vec   # reemplazar por valor promedio de vecinos
    return y
def corregir_extremos(y):
    y = y.copy()
    y[:20]  = y[21]          # igual al vecino
    y[-30::] = y[-31]
    return y



def desplazar_en_frecuencia(Xf, Fs, f0):
    """
    Xf: señal en frecuencia (array complejo)
    Fs: frecuencia de muestreo (Hz)
    f0: desplazamiento en frecuencia (Hz)
    """
    N = len(Xf)
    
    # Volver a tiempo
    xt = np.fft.ifft(Xf)

    # Crear vector de tiempo
    n = np.arange(N)
    t = n / Fs
    w= 2 * np.pi * f0 # rad/s
    #w=deg_Def*(np.pi/180)

    # Modulación para desplazar en frecuencia
    xt_shift = xt * np.exp(1j * w* t)
    #soft = suavizar_outliers(xt_shift)

    # Regresar a frecuencia
    Xf_shift = np.fft.fft(xt_shift)
    
    # --- Corrección de picos absurdos ---
    #Xf_shift = suavizar_picos(np.abs(Xf_shift), umbral_factor=8)
    Xf_shift = corregir_extremos(Xf_shift)
    

    return Xf_shift

def min_max(arr):
    print(10*np.log10(min(arr)),10*np.log10(max(arr)))
    return 
def min_max2(arr):
    print((min(arr)),(max(arr)))
    return 
def min_max3(arr):
    print(10*np.log10(1-min(arr)),10*np.log10(1-max(arr)))
    return 

def rectas(x,y,n,b):
    # Ajuste lineal y = m*x + b
    m, a = np.polyfit(x, y, 1)
    b_pow=1-pow(10,b/10)

    #print("Pendiente m =", m)
    #print("Intercepto a =", a)

    # Curva ajustada
    x_fit = np.linspace(min(x), max(x), n)
    y_fit = (m-1e-5) * x_fit + (a + b_pow)
    return x_fit,y_fit


def datos_componente3(nombre_archivo,n,b):
    data = []
    with open(nombre_archivo, 'r') as file: 
        csv_reader = csv.reader(file, quotechar='"')
        for row in csv_reader:
            aux = []
            for i in range(0, len(row)):
                aux.append(float(row[i]))
            data.append(aux)
    file.close()
    
    freq=[]
    data_pow=[]
    data_db=[]
    for i in range(len(data)):
        freq.append(data[i][0])
        data_pow.append(1-pow(10,(-data[i][1]/10)))
        data_db.append(data[i][1])
    
    #print('base_pow',np.mean(data_pow))
    #print('base_db',np.mean(data_db))
        
    xdup_freq = freq
    ydup_dat = data_pow
    
    x_fit,y_fit=rectas(xdup_freq,ydup_dat,n,b)
    #print('pow',np.mean(y_fit))
    #print('db',np.mean(10*np.log10(1-y_fit)))
    return x_fit,y_fit
