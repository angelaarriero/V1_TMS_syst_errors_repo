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


def TMS_instrument_output(array1_tiempo,array4_Tsky,t_load_r):
    
    #########################
    total_FEM_atmos_cmb=[]
    
    total_I1_FEM_atmos_cmb=[]
    total_I2_FEM_atmos_cmb=[]

    
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

    #
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
    
    Q,W,E,R,Z,X,C,V,xx1,data_RL_W,data_RL_FH,data_RL_omt,data_IL_omt,data_IL_hyb,data_RL_hyb,data_gain_lna,N,M,data_RL_load,U,Qs,Ws,Es,Rs,noise_lna_amp_roomT,N1,N2,N3,N4,Un1,Un2,Un3,Un4,sfN1=datos_simulados_RI(n,f1_file,f2_file,f3_file,f4_file,f5_file,f6_file,f7_file,f8_file,f9_file,f10_file)
    
    freq_spec=xx1

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
    
    ################# Noise temperature of the LNAs
    Tn1_copy=U
    Tn2_copy=U
    Tn3_copy=U
    Tn4_copy=U
    print(np.mean(U))
    
    #"""""
    ##### zero signals
    ## WINDOW
    Z=np.ones(n)*pow(10,(-1000/10)) #rl
    R=(np.ones(n)) - (np.ones(n)*pow(10,((0)/10))) #il
    SPOw=np.zeros(n) # SPO window
    
    E= (np.ones(n)) - (np.ones(n)*pow(10,((0)/10)))  # il
    RLirf= np.ones(n)*pow(10,(-1000/10)) #rl
    SPOirf=np.zeros(n)
    
    M= np.ones(n)*pow(10,(-1000/10))#rl
    
    ### FH
    Wl=(np.ones(n)) - (np.ones(n)*pow(10,((0)/10))) ## IL
    Ws=(np.ones(n)) - (np.ones(n)*pow(10,((0)/10)))
    Vl=np.ones(n)*pow(10,(-1000/10)) ## RL
    Vs=np.ones(n)*pow(10,(-1000/10))
    
    ### OMT
    Xl= np.ones(n)*pow(10,(-1000/10)) #rl
    Xs= np.ones(n)*pow(10,(-1000/10))
    Ql= (np.ones(n)) - (np.ones(n)*pow(10,((0)/10)))#il
    Qs= (np.ones(n)) - (np.ones(n)*pow(10,((0)/10)))
    
    ### HYB
    C1= np.ones(n)*pow(10,(-1000/10)) # rl
    C2= np.ones(n)*pow(10,(-1000/10))
    Q1= (np.ones(n)) - (np.ones(n)*pow(10,((0)/10)))#il
    Q2= (np.ones(n)) - (np.ones(n)*pow(10,((0)/10)))#il
    
    RLlna1s=np.ones(n)*pow(10,(-1000/10))
    RLlna2s=np.ones(n)*pow(10,(-1000/10))
    RLlna3l=np.ones(n)*pow(10,(-1000/10))
    RLlna4l=np.ones(n)*pow(10,(-1000/10))

    #Tn1_copy=np.zeros(n) 
    #Tn2_copy=np.zeros(n)
    #Tn3_copy=np.zeros(n)
    #Tn4_copy=np.zeros(n)

    #"""""
    
    """""

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
    
    #G1=N1_shifted
    #G2=N2_shifted
    #G3=N3_shifted
    #G4=N4_shifted
    
    # LNA GAIN
    #G1=N
    #G2=N
    #G3=N1_shifted
    #G4=N1_shifted

    ## lna noise
    
    U1=np.ones(n)*4.164
    U2=np.ones(n)*4.164
    U3=np.ones(n)*4.164
    U4=np.ones(n)*4.164
    
    #U1=np.array(Un3)
    #U2=np.array(Un4)
    #U3=np.array(Un3)
    #U4=np.array(Un4)
    
    U1=U1_shifted
    U2=U2_shifted
    U3=U3_shifted
    U4=U4_shifted
    
    GainBEM=np.ones(n)*923

    
    TnBem_var=np.ones(n)*454
    GainDC= np.ones(n)*43
    TnDC=np.ones(n)*230
    
    #SPOw=np.zeros(n) # SPO window
    #SPOirf=np.zeros(n) # SPO IR filter
    #SPO=np.zeros(n)
    
    """""

    ########################################################################

    ########Losses h=(1-RL)(1-IL) FEM 
    hw=(np.ones(n)-Z)*(np.ones(n)-R)*(np.ones(n)-SPOw) # h Window
    print('hw:',np.mean(hw))
    hirf=(np.ones(n)-RLirf)*(np.ones(n)-E)*(np.ones(n)-SPOirf) # h IR filter
    print('hirf:',np.mean(hirf))
    hfhs=(np.ones(n)-Vs)*(np.ones(n)-Ws) # h sky feed-horn
    print('hfhs:',np.mean(hfhs))
    homts=(np.ones(n)-Xs)*(np.ones(n)-Qs) # h sky OMT
    print('homts:',np.mean(homts))
    hfhl=(np.ones(n)-Vl)*(np.ones(n)-Wl) # h Load feed-horn
    print('hfhl:',np.mean(hfhl))
    homtl=(np.ones(n)-Xl)*(np.ones(n)-Ql) # h load omt
    print('homtl:',np.mean(homtl))
    hhyb1=(np.ones(n)-C1)*(np.ones(n)-Q1) # h hybrid X
    print('hhyb1:',np.mean(hhyb1))
    hhyb2=(np.ones(n)-C2)*(np.ones(n)-Q2) # h hybrid Y
    print('hhyb2:',np.mean(hhyb2))
    hlna1=(1-RLlna1s) # h LNA1
    print('hlna1:',np.mean(hlna1))
    hlna2=(1-RLlna2s) # h LNA2
    print('hlna2:',np.mean(hlna2))
    hlna3=(1-RLlna3l) # h LNA3
    print('hlna3:',np.mean(hlna3))
    hlna4=(1-RLlna4l) # h LNA4
    print('hlna4:',np.mean(hlna4))
    hload=(np.ones(n)-M)*(1-SPO) #h Cold-load
    print('hload:',np.mean(hload))

    

    ##### Summatory for effective Insertion losses 
    sum_tem_IL_sky=(Tw*R*hirf*hfhs*homts*0.5*(1-SPOw))+ (Tirf*E*hfhs*homts*0.5*(1-SPOirf))+(Tfhs*Ws*homts*0.5)+(Tomts*Qs)
    sum_tem_IL_load=(Tfhl*Wl*homtl*0.5)+(Tomtl*Ql)

    ##### Summatory for effective Return losses 
    sum_R_sky=(Z*hirf*hfhs*homts*0.5*(Tenv1/Tenv2)*(1-SPOw))+(RLirf*hfhs*homts*0.5*(1-SPOirf))+(Vs*homts*0.5)+(Xs)
    sum_R_load=(M*(1-SPO)*hfhl*homtl*0.5)+(Vl*homtl*0.5)+(Xl)

    ########################

    ####################
    
    ### effective SPO 
    a8=hirf*hfhs*homts*0.5*(SPOw)####---->>spo SKY
    a9=hfhs*homts*0.5*(SPOirf) ####---->>spo SKY
    a10=hfhl*homtl*0.5*(SPO) ####---->>spo LOAD

    #### Offsets
    offs1=((sum_tem_IL_sky)+(sum_tem_IL_load))+Tenv2*((sum_R_sky)+sum_R_load)+((T_ext_cry*a8+Tcryo1*a9)+(Tcryo2*a10))

    offl1=((sum_tem_IL_sky)-(sum_tem_IL_load))+Tenv2*((sum_R_sky)-sum_R_load)+((T_ext_cry*a8+Tcryo1*a9)-(Tcryo2*a10))
    offs2=((sum_tem_IL_sky)+(sum_tem_IL_load))+Tenv2*((sum_R_sky)+sum_R_load)+((T_ext_cry*a8+Tcryo1*a9)+(Tcryo2*a10))
    offl2=((sum_tem_IL_sky)-(sum_tem_IL_load))+Tenv2*((sum_R_sky)-sum_R_load)+((T_ext_cry*a8+Tcryo1*a9)-(Tcryo2*a10))

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

    #"""""
    plt.figure()
    fig, axs = plt.subplots(2, 2, figsize=(15, 5), sharex=True)  # 2x2 cuadrícula de subplots
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
    axs[0].set_ylabel(r'$S_{21}\ [dB]$', fontsize=20)
    axs[0].legend(loc='best', fontsize=9, ncol=3)
    axs[0].set_ylim(-0.4, 0)
    axs[0].grid()

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
    axs[1].set_ylabel(r'$S_{11}\ [dB]$', fontsize=20)
    axs[1].legend(loc='best', fontsize=10,ncol=3)
    axs[1].set_ylim(-90, 0)
    axs[1].grid()

    ####################### Subplot 3 #############################
    axs[2].plot(xx1, 10*np.log10(G1), label='Gain 5K', color='tab:purple')
    axs[2].tick_params(labelsize=15)
    axs[2].set_ylabel(r'$Gain\ [dB]$', fontsize=20)
    axs[2].set_xlabel(r'Frequency [GHz]', fontsize=20)
    axs[2].set_ylim(20, 38)
    axs[2].legend(loc='best', fontsize=10)
    axs[2].grid()

    # Segundo eje Y para el subplot 3
    ax2 = axs[2].twinx()
    ax2.plot(xx1, U, label='Tn 5K', color='y')
    ax2.set_ylabel('Tn [K]', fontsize=20, color='k')
    ax2.tick_params(axis='y', labelsize=15, labelcolor='k')
    ax2.set_ylim(0, 10)
    ax2.legend(loc='best')

    ####################### Subplot 4 #############################
    #axs[3].plot(xx1, 10*np.log10(GainBEM), color='r', label='Gain BEM')
    #axs[3].plot(xx1, 10*np.log10(GainDC), color='b', label='Gain DC')
    #axs[3].plot(xx1, 10*np.log10(GainDC), color='b', label='Gain DC')
    axs[3].tick_params(labelsize=15)
    axs[3].set_ylabel(r'$Gain\ [dB]$', fontsize=20)
    axs[3].set_xlabel(r'Frequency [GHz]', fontsize=20)
    axs[3].legend(loc='best', fontsize=10)
    axs[3].set_ylim(10, 35)
    axs[3].grid()
    
    # Segundo eje Y para el subplot 4
    ax3 = axs[3].twinx()
    #ax3.plot(xx1, TnBem_var, label='Tn BEM', color='y')
    #ax3.plot(xx1, TnDC, label='Tn DC', color='g')
    ax3.set_ylabel('Tn [K]', fontsize=20, color='k')
    ax3.tick_params(axis='y', labelsize=15, labelcolor='k')
    ax3.set_ylim(100, 500)
    #ax3.set_ylim(0, 10)
    ax3.legend(loc='best')

    # Ajustar espaciado automático
    plt.tight_layout()
    #plt.title(r'$\rm Optical \ and \ RF \ components \ vs \ Frequency$', fontsize=20)
    #plt.savefig('results_files_2/characteristic_inputs.pdf')
    #plt.savefig('plots_paper/relative_condi.pdf')
    #plt.savefig('results_files_2/imagenes_cap2/test_31_oct/nominal.png')
    #"""""

    ####################################################################

    for i in range(np.size(array1_tiempo)):
        
        t_sky_r=array4_Tsky[i] #
        
        
        ###### BEM ###### 
        e1_copy=Aphi1*(t_sky_r*a2 +t_load_r*a3)+A1_off## A1
        e2_copy=Aphi2*(t_sky_r*a2 -t_load_r*a3)+A2_off## A2
        e3_copy=Aphi3*(t_sky_r*a2 +t_load_r*a3)+B1_off## B1
        e4_copy=Aphi4*(t_sky_r*a2 -t_load_r*a3)+B2_off## B2
        

        ##### HYBRID SECOND STAGE (DIGITAL HYBRIDS X and Y) inside the FPGA. THERE IS NOT HYBRID LOSSES
        C1r_copy=(((e1_copy*G1)+(e2_copy*G2))) #C1
        C2r_copy=(((e1_copy*G1)-(e2_copy*G2))) #C2
        D1r_copy=(((e3_copy*G3)+(e4_copy*G4))) #D1
        D2r_copy=(((e3_copy*G3)-(e4_copy*G4))) #D2
        
        ############## SKY AND LOAD RECONSTRUCTED. I add the noise of the LNAs FIGURE 4 GREEN LINE (paper angela)
        sky_gth_copy=(C1r_copy+D1r_copy+(Tn1_copy*G1)+(Tn2_copy*G2))/G1
        load_gth_copy=(C2r_copy+D2r_copy+(Tn3_copy*G3)+(Tn4_copy*G4))/G3
        
        total_I1_FEM_atmos_cmb.append(sky_gth_copy)
        total_I2_FEM_atmos_cmb.append(load_gth_copy)
        Total_noBem_dc=(sky_gth_copy-load_gth_copy)
        
        total_FEM_atmos_cmb.append(Total_noBem_dc)

  
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
    print(np.mean(gainw)+np.mean(gainirf)+np.mean(gainfhs)+np.mean(gainomts1)+np.mean(gainhyb2))
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
    print(np.mean(gainLoad)+np.mean(gainfhs)+np.mean(gainomts1)+np.mean(gainhyb2))
    print('################################')

    print('··············#########################..............................')
    print(f"$T_sky$: {np.mean(sky_gth_copy):.6f}",'\\')
    print(f"$T_load$: {np.mean(load_gth_copy):.6f}",'\\')
    print(f"$Delta_T$: {np.mean(Total_noBem_dc):.6f}",'\\')
    #print("$Delta_T$:",np.mean(Total_noBem_dc),'\\')
    print("$pp_Delta_T$:",np.ptp(Total_noBem_dc),'\\')
   
    
    return total_FEM_atmos_cmb,freq_spec




