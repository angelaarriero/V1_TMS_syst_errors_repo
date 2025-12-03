import sys
import os
# Ruta absoluta a tu proyecto principal
ruta_base = "/home/aarriero/Documents/Angela_cmb/four_year/"
# Agregar al sys.path (al inicio para prioridad)
sys.path.insert(0, ruta_base)
print(ruta_base)

import csv
import numpy as np

from numpy.fft import fft

sys.path.append(os.path.join(ruta_base, "general_documents/definitions_files"))

from convert_csv_files_inputsignals_tosimulate import datos_componente
from convert_csv_files_inputsignals_tosimulate import datos_componente2
from convert_csv_files_inputsignals_tosimulate import datos_componente3

from convert_csv_files_inputsignals_tosimulate import sparams_to_power
from convert_csv_files_inputsignals_tosimulate import datos_simulados_RI
from convert_csv_files_inputsignals_tosimulate import conversion_dc
from convert_csv_files_inputsignals_tosimulate import desplazar_en_frecuencia
from convert_csv_files_inputsignals_tosimulate import min_max
from convert_csv_files_inputsignals_tosimulate import min_max2
from convert_csv_files_inputsignals_tosimulate import min_max3


import toml
# Carpeta donde está tu archivo TOML
ruta_toml = os.path.join(ruta_base, "general_documents", "data_files")
# Nombre del archivo TOML
toml_name = "input_params_simulation_V1_def"
toml_file = os.path.join(ruta_toml, toml_name + ".toml")
print("Ruta completa del archivo:", toml_file)
# Cargar el archivo TOML
data = toml.load(toml_file)
print("Archivo cargado correctamente")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec


def TMS_instrument_output(Tsky,t_load_r):
    
    #########################
    total_FEM_atmos_cmb=[]
    total_BEM_atmos_cmb=[]
    total_I1_BEM_atmos_cmb=[]
    total_I2_BEM_atmos_cmb=[]
    total_I1_FEM_atmos_cmb=[]
    total_I2_FEM_atmos_cmb=[]

    total_C1_BEM_atmos_cmb=[]
    total_C2_BEM_atmos_cmb=[]
    total_D1_BEM_atmos_cmb=[]
    total_D2_BEM_atmos_cmb=[]
    freq_spec=[]
    dev_total_system=[]
    noise_DC_cal=[]
    
    Tlss_1=[]
    Tlss_2=[]
    Tlss_3=[]
    Tlss_4=[]
    Tloss_sky=[]
    Tloss_load=[]

    #####################################################################
    n=data["Sky"]["n"]
    ######### FRONT-END 
    Tw=np.ones(n)*data["Window"]["Temperature"] # T. window
    Tirf=np.ones(n)*data["IRfilter"]["Temperature"] # T. IR filter
    Tfhs= np.ones(n)*data["FeedHornSky"]["Temperature"] # T. sky feed-horn
    Tomts=np.ones(n)*data["OMTsky"]["Temperature"] # T. OMT sky
    ThX=np.ones(n)*data["HybridX"]["Temperature"]# T. Hybrid X

    Tl=np.ones(n)*data["4KCL"]["Temperature"] # T. Cold-Load
    Tfhl=np.ones(n)*data["FeedHornload"]["Temperature"] # T.load feed-horn
    Tomtl=np.ones(n)*data["OMTload"]["Temperature"] # T. OMT load
    ThY=np.ones(n)*data["HybridY"]["Temperature"] # T.Hybrid Y

    Tenv1=np.ones(n)*data["Environment"]["Tenv1"] # T. environment 1 
    Tenv2=np.ones(n)*data["Environment"]["Tenv2"] # T. environment 2
    Troom=np.ones(n)*data["Environment"]["Troom"] # T. cryos 1 (T.room)
    
    T_ext_cry=np.ones(n)*data["Environment"]["T_ext_cry"] # 
    T_BEM_filter=np.ones(n)*data["Environment"]["T_BEM_filter"] # 
    T_after_filt_BEM=np.ones(n)*data["Environment"]["T_after_filt_BEM"] # 
    T_after_ampl_DC=np.ones(n)*data["Environment"]["T_after_ampl_DC"] # 
    T_FPGA=np.ones(n)*data["Environment"]["T_FPGA"] #
    
    Tcryo1=np.ones(n)*data["Environment"]["Tcryo1"] # T. cryos 2 (1st stage)
    Tcryo2=np.ones(n)*data["Environment"]["Tcryo2"] # T. cryos 3  (2nd stage)

    SPOw=np.ones(n)*pow(10,data["SPO"]["Window"]/10) # SPO window
    SPOirf=np.ones(n)*pow(10,data["SPO"]["IRfilter"]/10)# SPO IR filter
    SPO=pow(10,np.ones(n)*data["SPO"]["4KCL"]) # SPO Cold-load
    
    

    RLirf=np.ones(n)*pow(10,(data["IRfilter"]["RL"]/10)) # Return Loss IR filter
    RLlna1s=np.ones(n)*pow(10,(data["LNA1"]["RL"]/10)) # Return Loss lna1
    RLlna2s=np.ones(n)*pow(10,(data["LNA2"]["RL"]/10)) # Return Loss lna2
    RLlna3l=np.ones(n)*pow(10,(data["LNA3"]["RL"]/10)) # Return Loss lna3
    RLlna4l=np.ones(n)*pow(10,(data["LNA4"]["RL"]/10)) # Return Loss lna4

    ##################################
    ##############  BACK-END 
    ## AMPLIFIERS BEM
    ### https://qpmw.com/product/amplifiers/low-noise/amlna-0120-01/#!prettyPhoto
    RlnaBEM=np.ones(n)*pow(10,(data["BEM_amp"]["RL"]/10)) # Return Loss 
    GainBEM=np.ones(n)*pow(10,(data["BEM_amp"]["Gain"]/10)) # Gain 
    TnBem=np.ones(n)*((pow(10,(data["BEM_amp"]["Fnoise"])/10)-1)*300) ### Noise Temperature (noise figure= 4dB)

    ### FILTER
    RfilBem=np.ones(n)*pow(10,(data["BEM_filter"]["RL"]/10)) # Return Loss
    IfilBem=np.ones(n)*pow(10,(data["BEM_filter"]["IL"]/10)) # Insertion Loss

    ########################################

    ##################  DOWN-CONVERTER

    ## MIXER
    ### https://www.mouser.com/datasheet/2/1030/MDB_24H_2b-1700725.pdf?srsltid=AfmBOoqOyqtMh0OO23Dcg78k4z348bNhYnHH7l0cSBzv10tH27Pgxidj
    T_mixer=np.ones(n)*data["Mixer"]["T_mixer"] # 
    Fmixer=(np.ones(n)*(pow(10,(data["Mixer"]["Fnoise"]/10))-1)) # noise figure
    Lmixer=np.ones(n)*(1-pow(10,(data["Mixer"]["Lmixer"]/10))) # Conversion Loss 
    LO=np.ones(n)*(data["Mixer"]["LO"]/10) #T. added due to the LO

    
    # ----- MIXER EQUATION ----
    
    #Tmixer_A1=T_mixer*(Fmixer-pow(10,(2/10))+Lmixer)+LO*T_after_ampl_DC ##----> check this
    Tmixer_A1=1000
    Tmixer_A2=Tmixer_A1
    Tmixer_B1=Tmixer_A1
    Tmixer_B2=Tmixer_A1
    
    #### AMPLIFIERS DC
    ###https://www.mouser.com/datasheet/2/1030/PHA_202_2b-1700733.pdf?srsltid=AfmBOoqVZ1XoPo0bhApVXruea505aW-8BDN38TfeyBMXNrZrS3EYFWaa
    RlnaDC=np.ones(n)*pow(10,(data["DC_amp"]["RL"]/10)) # Return Loss 
    TnDC=np.ones(n)*((pow(10,(data["DC_amp"]["Fnoise"])/10)-1)*300) # Noise Temperature
    GainDC=np.ones(n)*pow(10,(data["DC_amp"]["Gain"]/10)) # Gain

    ##### FILTER
    TfilDC=np.ones(n)*300  # T. filter == T.room
    IfilDC=np.ones(n)*pow(10,(data["DC_filter"]["IL"]/10)) # Insertion Loss
    RfilDC=np.ones(n)*pow(10,(data["DC_filter"]["RL"]/10)) # Return Loss

    ##########------------------------##################################
    ################################## INITIAL PARAMETERS - signals shown in figure 3 (paper angela)
    ##########------------------------##################################
    ruta_files_n = os.path.join(ruta_base, "general_documents", "data_files")
    f1_file = os.path.join(ruta_files_n , "rl_hyb.csv")
    f2_file = os.path.join(ruta_files_n , "IL_hyb.csv")
    f3_file = os.path.join(ruta_files_n , "IL_omt.csv")
    #f4_file = os.path.join(ruta_files_n , "RL_omt.csv")
    f4_file = os.path.join(ruta_files_n , "OMT_measure_R.csv")
    f5_file = os.path.join(ruta_files_n , "RL_feedhorn.csv")
    f6_file = os.path.join(ruta_files_n , "RL_window.csv")
    f7_file = os.path.join(ruta_files_n , "LNA_20C.csv")
    f8_file = os.path.join(ruta_files_n , "CR117_load_RL.csv")
    f9_file = os.path.join(ruta_files_n , "TN_20_C.csv")
    f10_file = os.path.join(ruta_files_n , "noise_lna_amp_roomT.csv")
    f11_file = os.path.join(ruta_files_n , "gain_lna_amp_roomT.csv")
    f12_file = os.path.join(ruta_files_n , "GAIN_DC.csv")
    f13_file = os.path.join(ruta_files_n , "Noise_figu_DC.csv")
    f14_file = os.path.join(ruta_files_n , "TMS_IR_filter_10_layers_IL_10-20GHz.csv")
    
    
    """""
    plt.figure()
    plt.plot(xx14,IL_irfilter)
    plt.figure()
    plt.plot(xx14,new_IL_irfilter)
    print(np.mean(new_IL_irfilter))
    plt.figure()
    plt.plot(xx14,new_IL_IRFILTER)
    plt.figure()
    plt.plot(xx14,IL_irfilter)
    plt.figure()
    plt.plot(xx14,new_IL_IRFILTER)
    """""
    

    Q,W,E,R,Z,X,C,V,xx1,data_RL_W,data_RL_FH,data_RL_omt,data_IL_omt,data_IL_hyb,data_RL_hyb,data_gain_lna,N,M,data_RL_load,U,Qs,Ws,Es,Rs,noise_lna_amp_roomT,N1,N2,N3,N4,Un1,Un2,Un3,Un4,sfN1=datos_simulados_RI(n,f1_file,f2_file,f3_file,f4_file,f5_file,f6_file,f7_file,f8_file,f9_file,f10_file)

    x_fit,y_fit_IRfilter_il =datos_componente3(f14_file,n,0)
    IL_irfilter_t=y_fit_IRfilter_il
    
    x_fit2,y_fit_window_il =datos_componente3(f14_file,n,-0.05)
    
    x_fit3,y_fit_Q_il =datos_componente3(f14_file,n,-0.1)
    x_fit3,y_fit_OMT_il =datos_componente3(f14_file,n,-0.35)#-0.35
    min_max3(y_fit_OMT_il)
    
    Q=y_fit_Q_il
    #R=Rs #### il Window
    R=y_fit_window_il
    
    E=IL_irfilter_t ## il IRfilter
    Wl=Q## il FH
    Ws=Q 
    Vl=V ## rl FH
    Vs=V 
    Xl=X ## RL OMT
    Xs=X 
    Ql=y_fit_OMT_il ## IL OMT
    Qs=y_fit_OMT_il 
    
    C1=C ## RL HYB
    Q1=Q ## IL HYB
    C2=C 
    Q2=Q 
    
    
    
    #### LNAs 
    ## https://lownoisefactory.com/wp-content/uploads/2022/03/lnf-lnc6_20c.pdf
    ### I assume same Tn and G to all the amplifiers
    
    G1=N # GAIN
    G2=N
    G3=N
    G4=N
    
    U1=U
    U2=U
    U3=U
    U4=U
    
    ###########################........................... MOVE THE SIGNAL OF GAIN in frequency
    
    N1_shifted = desplazar_en_frecuencia(N, n, -1)#-10
    N2_shifted = desplazar_en_frecuencia(N, n, -2)
    N3_shifted = desplazar_en_frecuencia(N, n, 1)
    N4_shifted = desplazar_en_frecuencia(N, n, 2) 
    """""
    plt.figure()
    plt.plot(xx1, np.abs(N1_shifted), label='1')
    plt.plot(xx1, np.abs(N2_shifted), label='2.2')
    plt.plot(xx1, np.abs(N3_shifted), label='3.5')
    plt.plot(xx1, np.abs(N4_shifted), label='4.3')
    plt.legend()
    plt.xlabel("Frecuencia (Hz)")
    #plt.ylabel("Magnitud")
    #plt.savefig('results_plots_V1/shift_Ndiff_lna_inputs3.pdf')
    """""
    ###########################..........................
    
    ###########################........................... MOVE THE SIGNAL OF GAIN in frequency
    
    U1_shifted = desplazar_en_frecuencia(U, n,-1 )
    U2_shifted = desplazar_en_frecuencia(U, n, -2)
    U3_shifted = desplazar_en_frecuencia(U, n, 1)
    U4_shifted = desplazar_en_frecuencia(U, n, 2)
    """""
    plt.figure()
    plt.plot(xx1, np.abs(U), label='Original')
    plt.plot(xx1, np.abs(U1_shifted), label='100')
    plt.plot(xx1, np.abs(U2_shifted), label='200')
    plt.plot(xx1, np.abs(U3_shifted), label='300')
    plt.plot(xx1, np.abs(U4_shifted), label='400')
    plt.legend()
    plt.xlabel("Frecuencia (Hz)")
    #plt.ylabel("Magnitud")
    #plt.savefig('results_plots_V1/shift_Udiff_lna_inputs2.pdf')
    
    
    
    
    U1=np.ones(n)*4.164
    U2=np.ones(n)*3.9
    U3=np.ones(n)*4.5
    U4=np.ones(n)*3.5
    
    
    U1=U2_shifted ## t noise
    U2=U3_shifted
    U3=U4_shifted
    U4=U1_shifted
    
    G1=N2_shifted ## gain
    G2=N3_shifted
    G3=N4_shifted
    G4=N1_shifted
    
    
    
    Un1=np.array(Un1)
    Un2=np.array(Un2)
    Un3=np.array(Un3)
    Un4=np.array(Un4)
    U1=Un1
    U2=Un2
    U3=Un4
    U4=Un3
    print('noise')
    print(np.mean(U),np.mean(U1),np.mean(U2),np.mean(U3),np.mean(U4))
    
    Un1=np.array(Un1)
    Un2=np.array(Un2)
    Un3=np.array(Un3)
    Un4=np.array(Un4)
    
    U1=Un1
    U2=Un2
    U3=Un4
    U4=Un3
    print('noise')
    print(np.mean(U),np.mean(U1),np.mean(U2),np.mean(U3),np.mean(U4))
    
    G1=N1
    G2=N3
    G3=N4
    G4=N2
    print('gains')
    print(10*np.log10(np.mean(N)),10*np.log10(np.mean(G1)),10*np.log10(np.mean(G2)),10*np.log10(np.mean(G3)),10*np.log10(np.mean(G4)))
    
    ###########################..........................
    Un1=np.array(Un1)
    Un2=np.array(Un2)
    Un3=np.array(Un3)
    Un4=np.array(Un4)
    
    U1=Un1
    U2=Un2
    U3=Un4
    U4=Un3
    print('noise')
    print(np.mean(U),np.mean(U1),np.mean(U2),np.mean(U3),np.mean(U4))
    
    G1=N1
    G2=N2
    G3=N4
    G4=N3
    print('gains')
    print(10*np.log10(np.mean(N)),10*np.log10(np.mean(G1)),10*np.log10(np.mean(G2)),10*np.log10(np.mean(G3)),10*np.log10(np.mean(G4)))
    print(10*np.log10(np.mean(N)),10*np.log10(np.mean(N1)),10*np.log10(np.mean(N2)),10*np.log10(np.mean(N3)),10*np.log10(np.mean(N4)))
    """""
    
    
    

        ############### BEM NOISE AND GAIN 
    
    ### esto es nuevo, por eso no da lo mismo que el paper, pongo la
    ### implementacion del comportamiento del LNA a temp ambiente para simular
    ### el posible ruido de los amplificadores del BEM, PARA QUE NO SEA PLANO
    #"""""
    
    TnBem_var=(noise_lna_amp_roomT-np.mean(noise_lna_amp_roomT))*(np.mean(TnBem)/np.mean(noise_lna_amp_roomT))+np.mean(TnBem)
    """""
    plt.figure()
    plt.plot(TnBem_var)
    print('TnBem_var:',np.mean(TnBem_var))
    """""

    
    freq11, data_res11,xdup_freq11,ydup_dat11,xx11,gain_lna_amp_roomT =datos_componente(f11_file,n) #gainLNAtroom
    new_gain_lna_troom=sparams_to_power(gain_lna_amp_roomT,n) ### used to simulate the gain in the BEM AMPL
    GainBEM=new_gain_lna_troom
    
     ################# DC NOISE AND GAIN
    
    freq12, data_res12,xdup_freq12,ydup_dat12,xx11,gain_amp_DC =datos_componente2(f12_file,n) #gainampl DC
    new_gain_DC_troom=sparams_to_power(gain_amp_DC,n) ###
    
    GainDC=new_gain_DC_troom
    TnDC_conve=conversion_dc(f13_file,n)
    TnDC=TnDC_conve
    
    #"""""
   
    """""
    print('wind:Z:rl',np.mean(10*np.log10(Z)))
    print('R',10*np.log10(1-np.mean(np.mean(R))))
    print('FH:W_il',10*np.log10(1-np.mean(np.mean(W))))
    print('FH:V:rl',np.mean(10*np.log10(V)))
    print('irf:Z:rl',np.mean(10*np.log10(RLirf)))
    print('irf_R',10*np.log10(1-np.mean(np.mean(E))))
    print(np.size(E))
    
    print('OMT:X:rl',np.mean(10*np.log10(X)))
    print('Q',10*np.log10(1-np.mean(np.mean(Q))))
    print('HYB:C:rl',np.mean(10*np.log10(C)))
    print('Load:M:rl',np.mean(10*np.log10(M)))
    print('GAIN LNA:N:',np.mean((N)))
    print('Noise LNA:U:',np.mean((U)))
    print('GAIN BEM:N:',np.mean((GainBEM)))
    print('Noise BEM:U:',np.mean((TnBem_var)))
    print('GAIN DC:N:',np.mean((GainDC)))
    print('Noise DC:U:',np.mean((TnDC)))
    print('Tmixer_A1',np.mean(Tmixer_A1))
    
    """""
    
    """""
    print('wind:Z:rl',(min_max(Z)))
    print('R',10*np.log10(1-np.mean(np.mean(R))))
    print('FH:W_il',10*np.log10(1-np.mean(np.mean(W))))
    print('FH:V:rl',(min_max(V)))
    print('irf:Z:rl',np.mean(10*np.log10(RLirf)))
    print('irf_R',10*np.log10(1-np.mean(np.mean(E))))
    
    
    print('OMT:X:rl',(min_max(X)))
    print('Q',10*np.log10(1-np.mean(np.mean(Q))))
    print('HYB:C:rl',(min_max(C)))
    
    print('GAIN LNA:N:',(min_max(N)))
    print('Noise LNA:U:',(min_max2(U)))
    
    print('GAIN BEM:N:',min_max((GainBEM)))
    print('Noise BEM:U:',min_max2((TnBem_var)))
    print('GAIN DC:N:',min_max((GainDC)))
    print('Noise DC:U:',min_max2((TnDC)))
    print('Tmixer_A1',np.mean(Tmixer_A1))
    """""
    
    
    
    """""
    #### comment
    ## WINDOW
    Z=np.ones(n)*pow(10,(-1000/10)) #rl
    R=(np.ones(n)) - (np.ones(n)*pow(10,((0)/10))) #il
    E= (np.ones(n)) - (np.ones(n)*pow(10,((0)/10))) # il
    RLirf= np.ones(n)*pow(10,(-1000/10))#rl
    
    M=np.zeros(n)
    
    SPOw=np.zeros(n) # SPO window
    SPOirf=np.zeros(n) # SPO IR filter
    SPO=np.zeros(n)
    
    ### FH
    
    Vl=np.ones(n)*pow(10,(-100/10)) ## RL
    Vs=np.ones(n)*pow(10,(-100/10))
    
    Xl= np.ones(n)*pow(10,(-100/10)) #rl
    Xs= np.ones(n)*pow(10,(-100/10))
      
    """""
    
    
    """""
    ## COMMENT
    print('C1: RL',np.mean(C1))
    print('Q1:IL',np.mean(Q1))
    
    
    ### HYB
    C1= np.ones(n)*0.0022938963775313657 # rl
    C2= np.ones(n)*0.0022938963775313657
    Q1= (np.ones(n)*0.025160021119330106)
    Q2= (np.ones(n)*0.025160021119330106)
    
    # LNA GAIN
    G1=np.ones(n)*1939
    G2=np.ones(n)*1939
    G3=np.ones(n)*1939
    G4=np.ones(n)*1939
    
    U1=np.ones(n)*4.164
    U2=np.ones(n)*4.164
    U3=np.ones(n)*4.164
    U4=np.ones(n)*4.164
    
    RLlna1s=np.ones(n)*pow(10,(-1000/10))
    RLlna2s=np.ones(n)*pow(10,(-1000/10))
    RLlna3l=np.ones(n)*pow(10,(-1000/10))
    RLlna4l=np.ones(n)*pow(10,(-1000/10))
    
    SPOw=np.zeros(n) # SPO window
    R=(np.ones(n)) - (np.ones(n)*pow(10,((0)/10))) #il
    """""
    
    

    #SPOirf=np.zeros(n) # SPO IR filter
    #print('Z: RL',np.mean(Z))
    #print('R:IL',np.mean(R))
    #print('SPOw',np.mean(SPOw))
    
    #GainBEM=np.ones(n)*923
    #Z=np.ones(n)*0.000715352828481489 #rl
    
    #Xl= np.ones(n)*pow(10,(-32.67/10)) #rl
    #Xs= np.ones(n)*pow(10,(-32.67/10))
    

    #R=(np.ones(n)*0.01384414741820192) #il
    
    ### HYB
    #C1= np.ones(n)*0.0022938963775313657 # rl
    #C2= np.ones(n)*0.0022938963775313657
    #Q1= (np.ones(n)*0.025160021119330106)
    #Q2= (np.ones(n)*0.025160021119330106)
    
    #Z=np.ones(n)*pow(10,(-1000/10)) #rl
    #R=(np.ones(n)) - (np.ones(n)*pow(10,((0)/10))) #il
    #SPOw=np.zeros(n) # SPO window
    
    

    
    
    """""
    plt.figure()
    plt.plot(xx1, np.abs(U), label='nominal',color='k')
    plt.plot(xx1, np.abs(Un1), label='U1',color='r')
    plt.plot(xx1, np.abs(Un2), label='U2',color='b')
    plt.plot(xx1, np.abs(Un3), label='U3',color='y')
    plt.plot(xx1, np.abs(Un4), label='U4',color='g')
    plt.legend()
    plt.xlabel("Frecuencia (Hz)")
    #plt.ylabel("Magnitud")
    """""
    #Z=np.ones(n)*pow(10,(-34.9/10)) #rl
    #RLirf= np.ones(n)*pow(10,(-25/10))#rl
    #Xl= np.ones(n)*pow(10,(-32.67/10)) #rl
    #Xs= np.ones(n)*pow(10,(-32.67/10))
    #C1= np.ones(n)*pow(10,(-29.09/10)) # rl
    #C2= np.ones(n)*pow(10,(-29.09/10))

    """""

    ##### flat signals
    ## WINDOW
    Z=np.ones(n)*pow(10,(-34.9/10)) #rl
    R=(np.ones(n)) - (np.ones(n)*pow(10,((-0.0096)/10))) #il
    
    E= (np.ones(n)) - (np.ones(n)*pow(10,((-0.0034)/10))) # il
    RLirf= np.ones(n)*pow(10,(-25/10))#rl
    
    M= np.ones(n)*pow(10,(-56.73/10))#rl
    
    ### FH
    Wl=(np.ones(n)) - (np.ones(n)*pow(10,((-0.048)/10))) ## IL
    Ws=(np.ones(n)) - (np.ones(n)*pow(10,((-0.048)/10)))
    Vl=np.ones(n)*pow(10,(-37/10)) ## RL
    Vs=np.ones(n)*pow(10,(-37/10))
    
    ### OMT
    Xl= np.ones(n)*pow(10,(-32.67/10)) #rl
    Xs= np.ones(n)*pow(10,(-32.67/10))
    Ql= (np.ones(n)) - (np.ones(n)*pow(10,((-0.096)/10)))#il
    Qs= (np.ones(n)) - (np.ones(n)*pow(10,((-0.096)/10)))
    
    
    ### HYB
    C1= np.ones(n)*pow(10,(-29.09/10)) # rl
    C2= np.ones(n)*pow(10,(-29.09/10))
    Q1= (np.ones(n)) - (np.ones(n)*pow(10,((-0.096)/10)))#il
    Q2= (np.ones(n)) - (np.ones(n)*pow(10,((-0.096)/10)))#il
    
    
    ## RL 4kcl
    M=np.ones(n)*pow(10,(-56.73/10))
    
    # LNA GAIN
    G1=np.ones(n)*1939
    G2=np.ones(n)*1939
    G3=np.ones(n)*1939
    G4=np.ones(n)*1939
    
    # LNA GAIN
    
    #G1=N4
    #G2=N4
    #G3=N3
    #G4=N3
    
    
    
    # LNA GAIN
    #G1=N
    #G2=N
    #G3=N1_shifted
    #G4=N1_shifted

    ## lna noise
    
    #U1=np.ones(n)*4.164
    #U2=np.ones(n)*4.164
    #U3=np.ones(n)*4.164
    #U4=np.ones(n)*4.164
    
    #U1=np.array(Un3)
    #U2=np.array(Un4)
    #U3=np.array(Un3)
    #U4=np.array(Un4)
    
    
    
    GainBEM=np.ones(n)*923

    
    TnBem_var=np.ones(n)*454
    GainDC= np.ones(n)*43
    TnDC=np.ones(n)*230
    
    #SPOw=np.zeros(n) # SPO window
    #SPOirf=np.zeros(n) # SPO IR filter
    #SPO=np.zeros(n)
    
    """""
    
    #TnBem_var=TnBem
    TnBem_var1=TnBem_var
    TnBem_var2=TnBem_var
    TnBem_var3=TnBem_var
    TnBem_var4=TnBem_var
    
    """""
    plt.figure()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(xx1,C1) ## IL
    axes[1].plot(xx1,Q1) ## RL
    plt.savefig('results_plots_V1/hybs_inputs.pdf')
    plt.figure()
    fig, axes = plt.subplots(figsize=(10, 5))
    axes.plot(xx1,GainDC) ## IL
    #plt.savefig('results_plots_V1/DC_gain_inputs.pdf')
    
    """""
    

    """""
    plt.figure()
    
    plt.plot(xx1,sfN1,color='r')
    plt.plot(xx1,N,color='k')
    plt.figure()
    plt.plot(xx1,G1,color='k',linestyle='-',label='G1')
    plt.plot(xx1,G2,color='b',linestyle='-.',label='G2')
    plt.plot(xx1,G3,color='r',linestyle='-',label='G3')
    plt.plot(xx1,G4,color='m',linestyle='-.',label='G4')
    plt.legend(loc='best', fontsize=11, ncol=2)
    
    
    plt.figure()
    plt.plot(xx1,U,color='k')
    plt.figure()
    plt.plot(xx1,U1,color='k',linestyle='-',label='U1')
    plt.plot(xx1,U2,color='b',linestyle='-.',label='U2')
    plt.plot(xx1,U3,color='r',linestyle='-',label='U3')
    plt.plot(xx1,U4,color='m',linestyle='-.',label='U4')
    plt.legend(loc='best', fontsize=11, ncol=2)
    #plt.savefig('results_plots_V1/ampl_N1_N3_lna_inputs2.pdf')
    """""
    
    ########################################################################

    ########Losses h=(1-RL)(1-IL) FEM 
    hw=(np.ones(n)-Z)*(np.ones(n)-R)*(np.ones(n)-SPOw) # h Window
    #print('hw:',np.mean(hw))
    hirf=(np.ones(n)-RLirf)*(np.ones(n)-E)*(np.ones(n)-SPOirf) # h IR filter
    #print('hirf:',np.mean(hirf))
    hfhs=(np.ones(n)-Vs)*(np.ones(n)-Ws) # h sky feed-horn
    #print('hfhs:',np.mean(hfhs))
    homts=(np.ones(n)-Xs)*(np.ones(n)-Qs) # h sky OMT
    hfhl=(np.ones(n)-Vl)*(np.ones(n)-Wl) # h Load feed-horn
    #print('hfhl:',np.mean(hfhl))
    homtl=(np.ones(n)-Xl)*(np.ones(n)-Ql) # h load omt
    hhyb1=(np.ones(n)-C1)*(np.ones(n)-Q1) # h hybrid X
    hhyb2=(np.ones(n)-C2)*(np.ones(n)-Q2) # h hybrid Y
    hlna1=(1-RLlna1s) # h LNA1
    hlna2=(1-RLlna2s) # h LNA2
    hlna3=(1-RLlna3l) # h LNA3
    hlna4=(1-RLlna4l) # h LNA4
    hload=(np.ones(n)-M)*(1-SPO) #h Cold-load

    ################## BEM
    hlnaBEM=(np.ones(n)-RlnaBEM)
    hfilterBEM=np.ones(n)*((np.ones(n)-RfilBem)*(np.ones(n)-IfilBem)) ### transmission loss <0.5dB
    hlnaDC=(np.ones(n)-RlnaDC)
    hfilDC=(np.ones(n)-RfilDC)*(np.ones(n)-IfilDC)

    ##### Summatory for effective Insertion losses 
    sum_tem_IL_sky=(Tw*R*hirf*hfhs*homts*0.5*(1-SPOw))+ (Tirf*E*hfhs*homts*0.5*(1-SPOirf))+(Tfhs*Ws*homts*0.5)+(Tomts*Qs)
    sum_tem_IL_load=(Tfhl*Wl*homtl*0.5)+(Tomtl*Ql)
    """""
    print('...............sum_tem_IL_sky.........................')
    print('(Tw*R*hirf*hfhs*homts*0.5*(1-SPOw))',np.mean(Tw*R*hirf*hfhs*homts*0.5*(1-SPOw)))
    print('(Tirf*E*hfhs*homts*0.5*(1-SPOirf))',np.mean(Tirf*E*hfhs*homts*0.5*(1-SPOirf)))
    print('(Tfhs*Ws*homts*0.5)',np.mean((Tfhs*Ws*homts*0.5)))
    print('(Tomts*Qs)',np.mean((Tomts*Qs)))
    print('...............sum_tem_IL_load....................')
    print('(Tfhl*Wl*homtl*0.5)',np.mean((Tfhl*Wl*homtl*0.5)))
    print('(Tomtl*Ql)',np.mean((Tomtl*Ql)))
    """""
    ##### Summatory for effective Return losses 
    sum_R_sky=(Z*hirf*hfhs*homts*0.5*(Tenv1/Tenv2)*(1-SPOw))+(RLirf*hfhs*homts*0.5*(1-SPOirf))+(Vs*homts*0.5)+(Xs)
    sum_R_load=(M*(1-SPO)*hfhl*homtl*0.5)+(Vl*homtl*0.5)+(Xl)
    """""
    print('............sum_R_sky...................')
    print('(Z*hirf*hfhs*homts*0.5*(Tenv1/Tenv2)*(1-SPOw))',np.mean(Z*hirf*hfhs*homts*0.5*(Tenv1/Tenv2)*(1-SPOw)))
    print('(RLirf*hfhs*homts*0.5*(1-SPOirf))',np.mean(RLirf*hfhs*homts*0.5*(1-SPOirf)))
    print('(Vs*homts*0.5)',np.mean((Vs*homts*0.5)))
    print('(Xs)',np.mean((Xs)))
    print('............sum_R_load.....................')
    print('(M*(1-SPO)*hfhl*homtl*0.5)',np.mean(M*(1-SPO)*hfhl*homtl*0.5))
    print('(Vl*homtl*0.5)',np.mean((Vl*homtl*0.5)))
    print('(Xl)(Xl)',np.mean((Xl)))
    """""
    ########################

    ####################
    
    ### effective SPO 
    a8=hirf*hfhs*homts*0.5*(SPOw)####---->>spo SKY
    a9=hfhs*homts*0.5*(SPOirf) ####---->>spo SKY
    a10=hfhl*homtl*0.5*(SPO) ####---->>spo LOAD
    """""
    print('a8',np.mean(a8))
    print('a9',np.mean(a9))
    print('a10',np.mean(a10))
    """""

    #### Offsets
    offs1=((sum_tem_IL_sky)+(sum_tem_IL_load))+Tenv2*((sum_R_sky)+sum_R_load)+((T_ext_cry*a8+Tcryo1*a9)+(Tcryo2*a10))
    """""
    print('(sum_tem_IL_sky)',np.mean(sum_tem_IL_sky))
    print('(sum_tem_IL_load)',np.mean(sum_tem_IL_load))
    print('Tenv2*((sum_R_sky)+sum_R_load)',np.mean(Tenv2*((sum_R_sky)+sum_R_load)))
    """""
    offl1=((sum_tem_IL_sky)-(sum_tem_IL_load))+Tenv2*((sum_R_sky)-sum_R_load)+((T_ext_cry*a8+Tcryo1*a9)-(Tcryo2*a10))
    offs2=((sum_tem_IL_sky)+(sum_tem_IL_load))+Tenv2*((sum_R_sky)+sum_R_load)+((T_ext_cry*a8+Tcryo1*a9)+(Tcryo2*a10))
    offl2=((sum_tem_IL_sky)-(sum_tem_IL_load))+Tenv2*((sum_R_sky)-sum_R_load)+((T_ext_cry*a8+Tcryo1*a9)-(Tcryo2*a10))
    """""
    print('(T_ext_cry*a8+Tcryo2*a9)+(Tcryo3*a10)',np.mean((T_ext_cry*a8+Tcryo1*a9)+(Tcryo2*a10)))
    print('offs1',np.mean(offs1))
    print('offl1',np.mean(offl1))
    print('offs2',np.mean(offs2))
    print('offl2',np.mean(offl2))
    """""
    ### hybrid FIRST STAGE
    
    
    hyb_effect1=(ThX*Q1)+(Tenv2*C1) # (T.hybX* Il.HybX)+ (T.env2*RL.hybX)
    hyb_effect2=(ThY*Q2)+(Tenv2*C2) # (T.hybY* Il.HybY)+ (T.env2*RL.hybY)

    ######################### betas effective ########
    ### beta SKY and beta LOAD
    a2=hw*hirf*hfhs*homts*0.5  #### beta SKY
    a3=hfhl*homtl*0.5*(np.ones(n)-M)*(1-SPO) #### beta LOAD
    #print('(1-SPO)',np.mean((1-SPO)))
    ### effective losses of hybrids and LNAs BEM stage
    Aphi1=((hhyb1*hlna1)/(2*np.ones(n)))
    Aphi2=((hhyb1*hlna2)/(2*np.ones(n)))
    Aphi3=((hhyb2*hlna3)/(2*np.ones(n)))
    Aphi4=((hhyb2*hlna4)/(2*np.ones(n)))
    ######################################################################################

    #"""""
    A1_off=((Aphi1*offs1)+(hyb_effect1*hlna1)+(Tenv2*RLlna1s))
    B1_off=((Aphi3*offs2)+(hyb_effect2*hlna3)+(Tenv2*RLlna3l))
    A2_off=((Aphi2*offl1)+(hyb_effect1*hlna2)+(Tenv2*RLlna2s))
    B2_off=((Aphi4*offl2)+(hyb_effect2*hlna4)+(Tenv2*RLlna4l))

    Toff_total=(2*A2_off)+(2*B2_off)
    
    """""
    print('A1_off',np.mean(A1_off))
    print('B1_off',np.mean(B1_off))
    print('A2_off',np.mean(A2_off))
    print('B2_off',np.mean(B2_off))
    """""
    #########################################################################################

    
    #"""""
    HDC=(hlnaDC*hfilDC)*GainDC ## GAIN DONW-CONVERTER
    ## GAIN BEM
    loss1_Hs=hlnaBEM*hfilterBEM*G1*GainBEM*HDC
    loss2_Hs=hlnaBEM*hfilterBEM*G2*GainBEM*HDC
    loss3_Hs=hlnaBEM*hfilterBEM*G3*GainBEM*HDC
    loss4_Hs=hlnaBEM*hfilterBEM*G4*GainBEM*HDC
    """""
    print('loss1_Hs',np.mean(loss1_Hs))
    print('loss2_Hs',np.mean(loss2_Hs))
    print('loss3_Hs',np.mean(loss3_Hs))
    print('loss4_Hs',np.mean(loss4_Hs))
    """""

    ### LOSSES DUE TO BEM COMPONENTS, AMPLIFIERS AND FILTERS
    Tloss2_1=((T_BEM_filter*RlnaBEM*hfilterBEM*GainBEM)+(T_BEM_filter*IfilBem)+(T_after_filt_BEM*RfilBem))*HDC
    ### LOSSES DUE TO DOWN-CONVERTER COMPONENTS, MIXER, AMPLIFIERS AND FILTERS
    Tloss3_1=(((1-RlnaDC)*hfilDC*GainDC)*Tmixer_A1)+(T_after_ampl_DC*RlnaDC*hfilDC*GainDC)+(TfilDC*IfilDC)+T_FPGA*RfilDC

    ######################## Now add the NOISE TEMPERATURE of the FEM, BEM and DC
    #######......####################
    #### NOISE TEMPERATURE FEM (LNAs) TIMES GAINS OF BEM AND DC
    Tn1=U1*hlnaBEM*hfilterBEM*hfilDC*hlnaDC*GainDC*GainBEM*G1
    Tn2=U2*hlnaBEM*hfilterBEM*hfilDC*hlnaDC*GainDC*GainBEM*G2
    Tn3=U3*hlnaBEM*hfilterBEM*hfilDC*hlnaDC*GainDC*GainBEM*G3
    Tn4=U4*hlnaBEM*hfilterBEM*hfilDC*hlnaDC*GainDC*GainBEM*G4
    """""
    print('Tn1',np.mean(Tn1))
    print('Tn2',np.mean(Tn2))
    print('Tn3',np.mean(Tn3))
    print('Tn4',np.mean(Tn4))
    """""
    #######......####################

    #### NOISE TEMPERATURE BEM (amplifiers)
    #Tn2bem_1=TnBem*hfilterBEM*hfilDC*hlnaDC*GainDC*GainBEM

    Tn2bem_1=TnBem_var1*hfilterBEM*hfilDC*hlnaDC*GainDC*GainBEM
    #######......####################
    #### NOISE TEMPERATURE DOWN-CONVERTER (amplifiers)
    Tn3DC_1=TnDC*hfilDC*GainDC

    ############################################.....................###########################
    ###### to obtain the total noise added, we use the losses calculated to the FEM and the
    ###### losses calculated to the BEM AND DC

    ## NOISE TEMPERATURE IN THE X BRANCH
    # FEM
    Tn1FEM_1=(Tn1+Tn2)
    #print('Tn1FEM_1',np.mean(Tn1FEM_1))
    # BEM
    TnBEM_1= (Tn2bem_1)+(Tn2bem_1) 
    #print('TnBEM_1',np.mean(TnBEM_1))
    # DOWN-CONV
    TnDC_1=(Tn3DC_1)+(Tn3DC_1)
    #print('TnDC_1',np.mean(TnDC_1))

    ## NOISE TEMPERATURE IN THE Y BRANCH
    # FEM
    Tn1FEM_3=(Tn3+Tn4)
    #print('Tn1FEM_3',np.mean(Tn1FEM_3))
    # BEM
    TnBEM_3=(Tn2bem_1)+(Tn2bem_1)
    #print('TnBEM_3',np.mean(TnBEM_3))
    # DOWN-CONV
    TnDC_3=(Tn3DC_1)+(Tn3DC_1) 
    #print('TnDC_3',np.mean(TnDC_3))


    ######################################### SKY AND LOAD AT THE OUTPUT OF TMS FEM+BEM+DC ##############
    #### TOTAL GAIN Gtot will be the LNA, BEM AND DC GAINS, also, the system will be affected by the losses of
    #### the filters of the BEM and DC about 2dB
    Gtot=hfilterBEM*hfilDC*GainDC*GainBEM*G1
    
    ##### T noise completo
    Tn_tot_sky=(Tn1FEM_1+TnBEM_1+TnDC_1)/Gtot
    Tn_tot_load=(Tn1FEM_3+TnBEM_3+TnDC_3)/Gtot
    #print('Tn_tot_sky',np.mean(Tn_tot_sky))
    #print('Tn_tot_load',np.mean(Tn_tot_load))
    
    ############################### betas sky and load #########################################
    ###### beta sky y beta load
    beta_sky1=(a2*hlna1*hhyb1*loss1_Hs)/(2*np.ones(n))
    beta_sky2=(a2*hlna2*hhyb1*loss1_Hs)/(2*np.ones(n))
    beta_sky3=(a2*hlna3*hhyb2*loss1_Hs)/(2*np.ones(n))
    beta_sky4=(a2*hlna4*hhyb2*loss1_Hs)/(2*np.ones(n))

    beta_load1=(a3*hlna1*hhyb1*loss1_Hs)/(2*np.ones(n))
    beta_load2=(a3*hlna2*hhyb1*loss1_Hs)/(2*np.ones(n))
    beta_load3=(a3*hlna3*hhyb2*loss1_Hs)/(2*np.ones(n))
    beta_load4=(a3*hlna4*hhyb2*loss1_Hs)/(2*np.ones(n))

    T_sky_bet_total=(2*beta_sky2+2*beta_sky4)/Gtot
    T_load_bet_total=(2*beta_load2+2*beta_load4)/Gtot
    
    #"""""
    plt.figure()
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    colors2 = ["#9AD1D4",  # aqua pastel
          "#F7BFB4",  # coral pastel
          "#D7C6E9",  # lilac soft
          "#B4E1C5"]  # green-sea pastel

    # === 1. Primer subplot: Ganancias G1–G4 ===
    axs[0].plot(xx1, 10*np.log10(N), label=r'$\rm Nominal = %.2f$' % (10*np.log10(np.mean(N))), color='k')
    axs[0].plot(xx1, 10*np.log10(G1), label=r'$\rm G1 = %.2f$' % (10*np.log10(np.mean(G1))), color=colors2[0])
    axs[0].plot(xx1, 10*np.log10(G2), label=r'$\rm G2 = %.2f$' % (10*np.log10(np.mean(G2))), color=colors2[1])
    axs[0].plot(xx1, 10*np.log10(G3), label=r'$\rm G3 = %.2f$' % (10*np.log10(np.mean(G3))), color=colors2[2])
    axs[0].plot(xx1, 10*np.log10(G4), label=r'$\rm G4 = %.2f$' % (10*np.log10(np.mean(G4))), color=colors2[3])

    axs[0].set_ylabel("Gain [dB]", fontsize=20)
    axs[0].legend(fontsize=9,loc='best',ncol=2)
    axs[0].tick_params(labelsize=15)
    #axs[0].grid(True)
    colors = ["#AEC6CF",  # pastel blue
          "#FFB347",  # pastel orange
          "#B39EB5",  # pastel purple
          "#77DD77"]  # pastel green

    # === 2. Segundo subplot: Temperaturas de ruido U1–U4 ===
    axs[1].plot(xx1, np.abs(U), label=r'$\rm Nominal = %.2f$' % ((np.mean(U))), color='k')
    axs[1].plot(xx1, np.abs(U1), label=r'$\rm N1 = %.2f$' % ((np.mean(U1))), color=colors[0])
    axs[1].plot(xx1, np.abs(U2), label=r'$\rm N2 = %.2f$' % ((np.mean(U2))), color=colors[1])
    axs[1].plot(xx1, np.abs(U3), label=r'$\rm N3 = %.2f$' % ((np.mean(U3))), color=colors[2])
    axs[1].plot(xx1, np.abs(U4), label=r'$\rm N4 = %.2f$' % ((np.mean(U4))), color=colors[3])

    axs[1].set_xlabel(r'$\nu \ [GHz]$', fontsize=20)
    axs[1].set_ylabel("N [K]", fontsize=20)
    axs[1].legend(fontsize=9,loc='best',ncol=2)
    #axs[1].grid(True)
    axs[1].tick_params(labelsize=15)
    plt.tight_layout()
    #plt.savefig('plots_paper/Gain_Noise_diff_SHIFT.pdf')
    #plt.savefig('plots_paper/Gain_Noise_diff_3.pdf')
    #"""""

    """""
    plt.figure()
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    colors2 = ["#9AD1D4",  # aqua pastel
          "#F7BFB4",  # coral pastel
          "#D7C6E9",  # lilac soft
          "#B4E1C5"]  # green-sea pastel

    # === 1. Primer subplot: Ganancias G1–G4 ===
    axs[0].plot(xx1, 10*np.log10(N), label=r'$\rm Nominal = 0\,Hz$', color='k')
    axs[0].plot(xx1, 10*np.log10(G1), label=r'$\rm G1 = -2\,Hz$', color=colors2[0])
    axs[0].plot(xx1, 10*np.log10(G2), label=r'$\rm G2 = 1\,Hz$' , color=colors2[1])
    axs[0].plot(xx1, 10*np.log10(G3), label=r'$\rm G3 = 2\,Hz$' , color=colors2[2])
    axs[0].plot(xx1, 10*np.log10(G4), label=r'$\rm G4 = -1\,Hz$', color=colors2[3])

    axs[0].set_ylabel("Gain [dB]", fontsize=20)
    axs[0].legend(fontsize=9,loc='best',ncol=2)
    axs[0].tick_params(labelsize=15)
    #axs[0].grid(True)
    colors = ["#AEC6CF",  # pastel blue
          "#FFB347",  # pastel orange
          "#B39EB5",  # pastel purple
          "#77DD77"]  # pastel green

    # === 2. Segundo subplot: Temperaturas de ruido U1–U4 ===
    axs[1].plot(xx1, np.abs(U), label=r'$\rm Nominal = 0\,Hz$' , color='k')
    axs[1].plot(xx1, np.abs(U1), label=r'$\rm N1 = -2\,Hz$ ' , color=colors[0])
    axs[1].plot(xx1, np.abs(U2), label=r'$\rm N2 = 1\,Hz$', color=colors[1])
    axs[1].plot(xx1, np.abs(U3), label=r'$\rm N3 = 2\,Hz$' , color=colors[2])
    axs[1].plot(xx1, np.abs(U4), label=r'$\rm N4 = -1\,Hz$' , color=colors[3])

    axs[1].set_xlabel(r'$\nu \ [GHz]$', fontsize=20)
    axs[1].set_ylabel("N [K]", fontsize=20)
    axs[1].legend(fontsize=9,loc='best',ncol=2)
    #axs[1].grid(True)
    axs[1].tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('plots_paper/Gain_Noise_diff_SHIFT.pdf')
    #plt.savefig('plots_paper/Gain_Noise_diff.pdf')
    """""
        
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)  # 2x2 cuadrícula de subplots
    axs = axs.flatten()  # <-- aplanar la matriz de ejes (4 subplots en una lista)

    ####################### Subplot S21 #############################
    
    axs[0].plot(xx1, 10*np.log10(np.ones(n)-Qs), label='Omt', color='k')
    axs[0].plot(xx1, 10*np.log10(np.ones(n)-Q1), label='Hybrid', color='c',linestyle='dashdot')
    axs[0].plot(xx1, 10*np.log10(np.ones(n)-Ws), label='FeedHorn', color='r',linestyle='dashed')
    #axs[0].plot(xx1, Rs, label='Window', color='b')
    axs[0].plot(xx1, 10*np.log10(np.ones(n)-R), label='Window', color='b')
    #axs[0].plot(xx1, Es, label='IRfil', color='tab:orange')
    axs[0].plot(xx1, 10*np.log10(np.ones(n)-IL_irfilter_t), label='IRfil', color='tab:orange')
    axs[0].tick_params(labelsize=15)
    axs[0].set_ylabel(r'$\rm S_{21}\ [dB]$', fontsize=20)
    axs[0].legend(loc='best', fontsize=9, ncol=5)
    axs[0].set_ylim(-0.4, 0.1)
    #axs[0].set_xlabel(r'$\rm \nu \ [GHz]$', fontsize=20)

    ####################### Subplot S11 #############################
    #axs[1].plot(xx1, data_RL_W, color='b', label='Window')
    axs[1].plot(xx1, 10*np.log10(Z), color='b', label='Window')
    #axs[1].plot(xx1, data_RL_FH, color='r', label='FeedHorn')
    axs[1].plot(xx1, 10*np.log10(Vs), color='r', label='FeedHorn')
    #axs[1].plot(xx1, data_RL_omt, color='k', linestyle='dashed', label='OMT')
    axs[1].plot(xx1, 10*np.log10(Xs), color='k', linestyle='dashed', label='OMT')
    #axs[1].plot(xx1, data_RL_hyb, color='c', label='Hybrid')
    axs[1].plot(xx1, 10*np.log10(C1), color='c', label='Hybrid')
    #axs[1].plot(xx1, data_RL_load, color='m', label='Load')
    axs[1].plot(xx1, 10*np.log10(M), color='m', label='Load')
    axs[1].tick_params(labelsize=15)
    axs[1].set_ylabel(r'$\rm S_{11}\ [dB]$', fontsize=20)
    axs[1].legend(loc='best', fontsize=10,ncol=5)
    axs[1].set_ylim(-90, 0)
    #axs[1].set_xlabel(r'$\rm \nu \ [GHz]$', fontsize=20)
    plt.tight_layout()
    #plt.savefig('plots_paper/relative_condi.pdf')
    
    plt.figure()
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)  # 2x2 cuadrícula de subplots
    axs = axs.flatten()  # <-- aplanar la matriz de ejes (4 subplots en una lista)
    ####################### Subplot 3 #############################
    axs[0].plot(xx1, 10*np.log10(G1), label=r'$\rm Gain_{LNA}$', color='tab:purple')
    axs[0].tick_params(labelsize=15)
    axs[0].set_ylabel(r'$\rm Gain\ [dB]$', fontsize=20)
    axs[0].set_xlabel(r'$\rm \nu \ [GHz]$', fontsize=20)
    axs[0].set_ylim(20, 38)
    axs[0].legend(loc='best', fontsize=10)
    #axs[0].grid()

    # Segundo eje Y para el subplot 3
    ax2 = axs[0].twinx()
    ax2.plot(xx1, U1, label=r'$\rm N_{LNA}$', color='y')
    ax2.set_ylabel('N [K]', fontsize=20, color='k')
    ax2.tick_params(axis='y', labelsize=15, labelcolor='k')
    ax2.set_ylim(0, 10)
    ax2.legend(loc='lower right')

    ####################### Subplot 4 #############################
    axs[1].plot(xx1, 10*np.log10(GainBEM), color='r', label=r'$\rm Gain_{BEM}$')
    axs[1].plot(xx1, 10*np.log10(GainDC), color='b', label=r'$\rm Gain_{DC}$')
    #axs[3].plot(xx1, 10*np.log10(GainDC), color='b', label='Gain DC')
    axs[1].tick_params(labelsize=15)
    axs[1].set_ylabel(r'$\rm Gain\ [dB]$', fontsize=20)
    axs[1].set_xlabel(r'$\rm \nu \ [GHz]$', fontsize=20)
    axs[1].legend(loc='best', fontsize=10)
    axs[1].set_ylim(10, 35)
    #axs[1].grid()
    
    # Segundo eje Y para el subplot 4
    ax3 = axs[1].twinx()
    ax3.plot(xx1, TnBem_var, label=r'$\rm N_{BEM}$', color='y')
    ax3.plot(xx1, TnDC, label=r'$\rm N_{DC}$', color='g')
    ax3.set_ylabel('N [K]', fontsize=20, color='k')
    ax3.tick_params(axis='y', labelsize=15, labelcolor='k')
    #ax3.set_ylim(100, 500)
    #ax3.set_ylim(0, 10)
    ax3.legend(loc='best')

    # Ajustar espaciado automático
    plt.tight_layout()
    #plt.savefig('plots_paper/relative_condi_2.pdf')
    
    
    t_sky_r=Tsky #
        
    ########################### deltaT equation
    e1=Aphi1*(t_sky_r*a2 +t_load_r*a3)+A1_off## A1
    e2=Aphi2*(t_sky_r*a2 -t_load_r*a3)+A2_off## A2
    e3=Aphi3*(t_sky_r*a2 +t_load_r*a3)+B1_off## B1
    e4=Aphi4*(t_sky_r*a2 -t_load_r*a3)+B2_off## B2
        
        
    
        
    ###### BEM ###### 
    e1_copy=Aphi1*(t_sky_r*a2 +t_load_r*a3)+A1_off## A1
    e2_copy=Aphi2*(t_sky_r*a2 -t_load_r*a3)+A2_off## A2
    e3_copy=Aphi3*(t_sky_r*a2 +t_load_r*a3)+B1_off## B1
    e4_copy=Aphi4*(t_sky_r*a2 -t_load_r*a3)+B2_off## B2
        
    ################# Noise temperature of the LNAs
    Tn1_copy=U1 
    Tn2_copy=U2
    Tn3_copy=U3
    Tn4_copy=U4
    ##### HYBRID SECOND STAGE (DIGITAL HYBRIDS X and Y) inside the FPGA. THERE IS NOT HYBRID LOSSES
    C1r_copy=(((e1_copy*G1)+(e2_copy*G2))) #C1
    C2r_copy=(((e1_copy*G1)-(e2_copy*G2))) #C2
    D1r_copy=(((e3_copy*G3)+(e4_copy*G4))) #D1
    D2r_copy=(((e3_copy*G3)-(e4_copy*G4))) #D2
       
    ############## SKY AND LOAD RECONSTRUCTED. I add the noise of the LNAs FIGURE 4 GREEN LINE (paper angela)
    sky_gth_copy=(C1r_copy+D1r_copy+(Tn1_copy*G1)+(Tn2_copy*G2))/G1
    load_gth_copy=(C2r_copy+D2r_copy+(Tn3_copy*G3)+(Tn4_copy*G4))/G3
        
    #total_I1_FEM_atmos_cmb.append(sky_gth_copy)
    #total_I2_FEM_atmos_cmb.append(load_gth_copy)
    
    total_I1_FEM_atmos_cmb=sky_gth_copy
    total_I2_FEM_atmos_cmb=load_gth_copy
    Total_noBem_dc=(sky_gth_copy-load_gth_copy)
        
    #total_FEM_atmos_cmb.append(Total_noBem_dc)
    total_FEM_atmos_cmb=Total_noBem_dc

        
    ###### ....... ###### 
        
    ##################### fem+bem+dc continuation
    #"""""
    s1=(loss1_Hs*e1)+(Tloss2_1)+(Tloss3_1) ### A1+BEM+DC
    s2=(loss2_Hs*e2)+(Tloss2_1)+(Tloss3_1) ### A2+BEM+DC
    s3=(loss3_Hs*e3)+(Tloss2_1)+(Tloss3_1) ### B1+BEM+DC
    s4=(loss4_Hs*e4)+(Tloss2_1)+(Tloss3_1) ### B2+BEM+DC
     #"""""
        
    #""""" in this part I remove the contribution of the beta sky and load, and also the input sky/load
    s1_noise_contr=(loss1_Hs*A1_off)+(Tloss2_1)+(Tloss3_1) ### A1+BEM+DC
    s2_noise_contr=(loss1_Hs*A2_off)+(Tloss2_1)+(Tloss3_1) ### A2+BEM+DC
    s3_noise_contr=(loss1_Hs*B1_off)+(Tloss2_1)+(Tloss3_1) ### B1+BEM+DC
    s4_noise_contr=(loss1_Hs*B2_off)+(Tloss2_1)+(Tloss3_1) ### B2+BEM+DC
        #""
        
    #Tlss_1.append(s1_noise_contr)
    #Tlss_2.append(s2_noise_contr)
    #Tlss_3.append(s3_noise_contr)
    #Tlss_4.append(s4_noise_contr)
    
    Tlss_1=s1_noise_contr
    Tlss_2=s2_noise_contr
    Tlss_3=s3_noise_contr
    Tlss_4=s4_noise_contr
        
    #""
    #"""""
    ##### second hybrid inside the FPGA
    C1r=(((s1)+(s2)))
    C2r=(((s1)-(s2)))
    D1r=(((s3)+(s4)))
    D2r=(((s3)-(s4)))
        
        
        
    #total_C1_BEM_atmos_cmb.append(C1r)
    #total_C2_BEM_atmos_cmb.append(C2r)
    #total_D1_BEM_atmos_cmb.append(D1r)
    #total_D2_BEM_atmos_cmb.append(D2r)
    
    total_C1_BEM_atmos_cmb=C1r
    total_C2_BEM_atmos_cmb=C2r
    total_D1_BEM_atmos_cmb=D1r
    total_D2_BEM_atmos_cmb=D2r
        #""""
        ###########################################################################################################
        ############# NOW HERE, WE SUMM THE SIGNALS C1,D1, C2 ,D2 WITH THE THERMAL NOISE ADDED BY THE AMPLIFIERS
        ################################# FEM+BEM+DC

    sky_gth2=(C1r+D1r+Tn1FEM_1+TnBEM_1+TnDC_1) ## Tsky
    load_gth2=(C2r+D2r+Tn1FEM_3+TnBEM_3+TnDC_3) ## Tload
        
    #total_I1_BEM_atmos_cmb.append(sky_gth2)
    #total_I2_BEM_atmos_cmb.append(load_gth2)
    
    total_I1_BEM_atmos_cmb=sky_gth2
    total_I2_BEM_atmos_cmb=load_gth2

    Total_bem_dc=(sky_gth2-load_gth2)/Gtot ##### WE OBTAIN THE OUTPUT DIVIDING BY THE TOTAL GAIN CONTRIBUTION
    #total_BEM_atmos_cmb.append(Total_bem_dc)
    total_BEM_atmos_cmb=Total_bem_dc
    #freq_spec.append(xx1)
    freq_spec=xx1
    

    ####################################################################

    
    """""    
    print('tsky,tload',np.mean(t_sky_r),np.mean(t_load_r))
    print('Aphi1',np.mean(Aphi1))
    print('a2',np.mean(a2))
    print('a3',np.mean(a3))
    print('A1_off',np.mean(A1_off))
        
    print('Aphi2',np.mean(Aphi2))
    print('A2_off',np.mean(A2_off))
        
    print('Aphi3',np.mean(Aphi3))
    print('B1_off',np.mean(B1_off))
        
    print('Aphi4',np.mean(Aphi4))
    print('B2_off',np.mean(B2_off))
       
        
    print('A1',np.mean(e1))
    print('A2',np.mean(e2))
    print('B1',np.mean(e3))
    print('B2',np.mean(e4))
    
    print('C1r',np.mean(C1r))
    print('C2r',np.mean(C2r))
    print('D1r',np.mean(D1r))
    print('D2r',np.mean(D2r))
    
    print('Tn1FEM_1+TnBEM_1+TnDC_1',np.mean(Tn1FEM_1+TnBEM_1+TnDC_1))
    print('Tn1FEM_3+TnBEM_3+TnDC_3',np.mean(Tn1FEM_3+TnBEM_3+TnDC_3))
    """""
    print('################## Component TEMPERATURE CONTRIBUTION')
    ## Branch 1 -->> sky
    gainw=((Tw)*R)*(1-SPOw)+T_ext_cry*SPOw
    print('R',np.mean(R),'1-SPOw',np.mean(1-SPOw),'T_ext_cry',np.mean(T_ext_cry),'SPOw',np.mean(SPOw))
    print('spoW',np.mean(SPOw))
    print('window',np.mean(gainw))
    print('window',np.std(gainw))
    gainirf=(Tirf*E*(1-SPOirf))+(Tcryo1*SPOirf)/(hw)
    print('Tirf',np.mean(Tirf),'E',np.mean(E),'1-SPOirf',np.mean(1-SPOirf),'Tcryo1',np.mean(Tcryo1),'SPOirf',np.mean(SPOirf),'hw',np.mean(hw))
    print('filter',np.mean(gainirf))
    print('filter',np.std(gainirf))
    gainfhs=((Tfhs)*Wl)/(hw*hirf)
    print('Tfhs',np.mean(Tfhs),'Wl',np.mean(Wl),'hw',np.mean(hw),'hirf',np.mean(hirf))
    print('feedhorn sky',np.mean(gainfhs))
    print('feedhorn sky',np.std(gainfhs))
    gainomts1=((Tomts)*Ql)/(hw*hirf*hfhs)
    print('Tomts',np.mean(Tomts),'Ql',np.mean(Ql),'hw',np.mean(hw),'hfhs',np.mean(hfhs))
    print('omt sky out1',np.mean(gainomts1))
    print('omt sky out1',np.std(gainomts1))
    gainhyb2=((ThY)*Q1)/(hw*hirf*hfhs*homts)
    print('hybrid sky out2',np.mean(gainhyb2))
    print('hybrid sky out2',np.std(gainhyb2))
    print('################################')
    sky_tot_gain=np.mean(gainw)+np.mean(gainirf)+np.mean(gainfhs)+np.mean(gainomts1)+np.mean(gainhyb2)
    print('sky_tot_gain',sky_tot_gain)
    print('sky_tot_gain*skybeta',sky_tot_gain*np.mean(T_sky_bet_total))
    print('################################')
    print('··············######  LOAD ##############..............................')
    gainLoad=(SPO*Tcryo2)
    print('gain_load',np.mean(gainLoad))
    print('gain_load',np.std(gainLoad))
    gainfhs=((Tfhs)*Wl)/(hload)
    print('feedhorn sky',np.mean(gainfhs))
    print('feedhorn sky',np.std(gainfhs))
    gainomts1=((Tomts)*Ql)/(hload*hfhl)
    print('omt sky out1',np.mean(gainomts1))
    print('omt sky out1',np.std(gainomts1))
    gainhyb2=((ThY)*Q1)/(hload*hfhl*homtl)
    print('hybrid sky out2',np.mean(gainhyb2))
    print('hybrid sky out2',np.std(gainhyb2))
    print('################################')
    load_tot_gain=np.mean(gainLoad)+np.mean(gainfhs)+np.mean(gainomts1)+np.mean(gainhyb2)
    print('load_tot_gain',load_tot_gain)
    print('load_tot_gain*loadbeta',load_tot_gain*np.mean(T_load_bet_total))

    print('suma sky + load',sky_tot_gain+load_tot_gain, np.std(sky_tot_gain+load_tot_gain))
    suma_betas_tot=(sky_tot_gain*np.mean(T_sky_bet_total))+(load_tot_gain*np.mean(T_load_bet_total))
    print('(suma sky + load)betas',(suma_betas_tot),np.std(suma_betas_tot))
    
    print('################################')


    print('··············#########################..............................')
    print(f"$T_sky$: {np.mean(sky_gth2/Gtot):.6f}",'\\')
    print(f"$T_load$: {np.mean(load_gth2/Gtot):.6f}",'\\')
    print("$Delta_T$:",np.mean(Total_bem_dc),'\\')
    print("$std_Delta_T$:",np.std(Total_bem_dc),'\\')
    print("$pp_Delta_T$:",np.ptp(Total_bem_dc),'\\')
    print('Gtot',np.mean(Gtot))
    
    return total_BEM_atmos_cmb,total_FEM_atmos_cmb,total_I1_BEM_atmos_cmb,total_I2_BEM_atmos_cmb,total_I1_FEM_atmos_cmb,total_I2_FEM_atmos_cmb,freq_spec,T_sky_bet_total,T_load_bet_total,Tlss_1,Tlss_2,Tlss_3,Tlss_4,Gtot,Tn_tot_sky,Tn_tot_load,V,dev_total_system,U1_shifted,U2_shifted,U3_shifted,U4_shifted,N1_shifted,N2_shifted,N3_shifted,N4_shifted

