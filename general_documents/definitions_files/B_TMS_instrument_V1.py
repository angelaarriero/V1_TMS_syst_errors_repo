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
    

    Q,W,E,R,Z,X,C,V,xx1,data_RL_W,data_RL_FH,data_RL_omt,data_IL_omt,data_IL_hyb,data_RL_hyb,data_gain_lna,N,M,data_RL_load,U,Qs,Ws,Es,Rs,noise_lna_amp_roomT,N1,N2,N3,N4,Un1,Un2,Un3,Un4,sfN1=datos_simulados_RI(n,f1_file,f2_file,f3_file,f4_file,f5_file,f6_file,f7_file,f8_file,f9_file,f10_file)

    x_fit,y_fit_IRfilter_il =datos_componente3(f14_file,n,0)
    IL_irfilter_t=y_fit_IRfilter_il
    
    x_fit2,y_fit_window_il =datos_componente3(f14_file,n,-0.05)
    
    x_fit3,y_fit_Q_il =datos_componente3(f14_file,n,-0.1)
    x_fit3,y_fit_OMT_il =datos_componente3(f14_file,n,-0.35)#-0.35
    #min_max3(y_fit_OMT_il)
    
    Q=y_fit_Q_il
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

    ###########################..........................
    
    ###########################........................... MOVE THE SIGNAL OF GAIN in frequency
    
    U1_shifted = desplazar_en_frecuencia(U, n,-1 )
    U2_shifted = desplazar_en_frecuencia(U, n, -2)
    U3_shifted = desplazar_en_frecuencia(U, n, 1)
    U4_shifted = desplazar_en_frecuencia(U, n, 2)

    ###########################..........................
        ############### BEM NOISE AND GAIN 
    
    ### esto es nuevo, por eso no da lo mismo que el paper, pongo la
    ### implementacion del comportamiento del LNA a temp ambiente para simular
    ### el posible ruido de los amplificadores del BEM, PARA QUE NO SEA PLANO
    #"""""
    
    TnBem_var=(noise_lna_amp_roomT-np.mean(noise_lna_amp_roomT))*(np.mean(TnBem)/np.mean(noise_lna_amp_roomT))+np.mean(TnBem)

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
    TnBem_var1=TnBem_var
    TnBem_var2=TnBem_var
    TnBem_var3=TnBem_var
    TnBem_var4=TnBem_var
    
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

    #########################################################################################

    
    #"""""
    HDC=(hlnaDC*hfilDC)*GainDC ## GAIN DONW-CONVERTER
    ## GAIN BEM
    loss1_Hs=hlnaBEM*hfilterBEM*G1*GainBEM*HDC
    loss2_Hs=hlnaBEM*hfilterBEM*G2*GainBEM*HDC
    loss3_Hs=hlnaBEM*hfilterBEM*G3*GainBEM*HDC
    loss4_Hs=hlnaBEM*hfilterBEM*G4*GainBEM*HDC

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

    #######......####################

    #### NOISE TEMPERATURE BEM (amplifiers)
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
    axs[1].plot(xx1, 10*np.log10(Z), color='b', label='Window')
    axs[1].plot(xx1, 10*np.log10(Vs), color='r', label='FeedHorn')
    axs[1].plot(xx1, 10*np.log10(Xs), color='k', linestyle='dashed', label='OMT')
    axs[1].plot(xx1, 10*np.log10(C1), color='c', label='Hybrid')
    axs[1].plot(xx1, 10*np.log10(M), color='m', label='Load')
    axs[1].tick_params(labelsize=15)
    axs[1].set_ylabel(r'$\rm S_{11}\ [dB]$', fontsize=20)
    axs[1].legend(loc='best', fontsize=10,ncol=5)
    axs[1].set_ylim(-90, 0)
    plt.tight_layout()
    plt.savefig('/home/aarriero/Documents/Angela_cmb/four_year/cap2_TOD_TMS/TOD_V_dic2025/B_results_plots/A_relative_condi.pdf')
    
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
    ax3.legend(loc='best')

    # Ajustar espaciado automático
    plt.tight_layout()
    plt.savefig('/home/aarriero/Documents/Angela_cmb/four_year/cap2_TOD_TMS/TOD_V_dic2025/B_results_plots/A_relative_condi_amplifiers.pdf')

    ####################################################################

    for i in range(np.size(array1_tiempo)):
        
        t_sky_r=array4_Tsky[i] #
        
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
        
        total_I1_FEM_atmos_cmb.append(sky_gth_copy)
        total_I2_FEM_atmos_cmb.append(load_gth_copy)
        Total_noBem_dc=(sky_gth_copy-load_gth_copy)
        
        total_FEM_atmos_cmb.append(Total_noBem_dc)

        
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
        
        Tlss_1.append(s1_noise_contr)
        Tlss_2.append(s2_noise_contr)
        Tlss_3.append(s3_noise_contr)
        Tlss_4.append(s4_noise_contr)
        
        #""
        #"""""
        ##### second hybrid inside the FPGA
        C1r=(((s1)+(s2)))
        C2r=(((s1)-(s2)))
        D1r=(((s3)+(s4)))
        D2r=(((s3)-(s4)))
        
        
        
        total_C1_BEM_atmos_cmb.append(C1r)
        total_C2_BEM_atmos_cmb.append(C2r)
        total_D1_BEM_atmos_cmb.append(D1r)
        total_D2_BEM_atmos_cmb.append(D2r)
        #""""
        ###########################################################################################################
        ############# NOW HERE, WE SUMM THE SIGNALS C1,D1, C2 ,D2 WITH THE THERMAL NOISE ADDED BY THE AMPLIFIERS
        ################################# FEM+BEM+DC

        sky_gth2=(C1r+D1r+Tn1FEM_1+TnBEM_1+TnDC_1) ## Tsky
        load_gth2=(C2r+D2r+Tn1FEM_3+TnBEM_3+TnDC_3) ## Tload
        
        total_I1_BEM_atmos_cmb.append(sky_gth2)
        total_I2_BEM_atmos_cmb.append(load_gth2)

        Total_bem_dc=(sky_gth2-load_gth2)/Gtot ##### WE OBTAIN THE OUTPUT DIVIDING BY THE TOTAL GAIN CONTRIBUTION
        total_BEM_atmos_cmb.append(Total_bem_dc)
        freq_spec.append(xx1)
        dev_total_system.append(Total_bem_dc-(t_sky_r-t_load_r))

    
    print('################################')


    print('··············#########################..............................')
    print(f"$T_sky$: {np.mean(sky_gth2/Gtot):.6f}",'\\')
    print(f"$T_load$: {np.mean(load_gth2/Gtot):.6f}",'\\')
    print("$Delta_T$:",np.mean(Total_bem_dc),'\\')
    print("$std_Delta_T$:",np.std(Total_bem_dc),'\\')
    print("$pp_Delta_T$:",np.ptp(Total_bem_dc),'\\')
    print('Gtot',np.mean(Gtot))
    
    return total_BEM_atmos_cmb,total_I1_BEM_atmos_cmb,total_I2_BEM_atmos_cmb,freq_spec,T_sky_bet_total,T_load_bet_total,Tlss_1,Tlss_2,Tlss_3,Tlss_4,Gtot,Tn_tot_sky,Tn_tot_load


def noise_instrument(array1_tiempo,freq_spec,total_BEM_atmos_cmb):
    ###############################################################################################
    ############################################### NOISE
    sigma_new=[]
    BEM_sigma_total=[]
    noise_total_sigma=[]
    Ti_total=[]

    ################### array1_loaded[i] tiempo en segundos ----- sigma(i)=Tsys/sqrt(frequ(i)*time)

    for i in range(np.size(array1_tiempo)):
        for j in range(np.size(freq_spec[0])):
            #print(total_BEM_atmos_cmb[i][j])
            #print(freq_spec[0][j])
            #print(array1_loaded[i])
            signoise    = total_BEM_atmos_cmb[i][j]/(np.sqrt(freq_spec[0][j]*array1_tiempo[i]))
            sigma_new.append(signoise)

    for i in range(np.size(sigma_new)):
        media = 0      
        sigma = sigma_new[i]     
        num_muestras = 100 
        #num_muestras = 1000 
        datos = np.random.normal(loc=media, scale=sigma, size=num_muestras)
        mean_data=np.mean(datos)
        noise_total_sigma.append(mean_data)

    Noise_total_time=np.reshape(noise_total_sigma, (np.size(array1_tiempo), np.size(freq_spec[0])))

    for i in range(np.size(array1_tiempo)):
        for j in range(np.size(freq_spec[0])):
            summ_Ti_ni    = total_BEM_atmos_cmb[i][j]+Noise_total_time[i][j]
            Ti_total.append(summ_Ti_ni)

    Ti_total=np.reshape(Ti_total, (np.size(array1_tiempo), np.size(freq_spec[0])))#### en este punto la señal ya tiene ruido asociado
    ### que viene dado por una señal random con sigma= tsys/sqrt(freq*tiempo) y media=0
    ########################################################################################################################
    ###################### SUBBANDAS PROMEDIO
    #############################################################################################
    #### aqui voy a calcular el promedio cada 250 Mhz con pesos == 1
    increase=0
    new_freq=freq_spec[0]*1e9
    band_pass_prom=[]
    size_arr_freq=int((new_freq[np.size(new_freq)-1]-new_freq[0])/250e6)
    index_band_pass=[]
    sub_band_def=[]
    for i in range(np.size(array1_tiempo)):### tiempo de observacion en segundos
        for j in range(size_arr_freq): ####### frecuencias de 10 a 20 Ghz
            x = new_freq[0]+(240e6*j)+(j*1e7) #### elijo la frecuencia principal (10Ghz)+ 240Mhz*(0)+(0)*1e7
            x_f = x+240e6
            sub_band_def.append(np.array([x,x_f]))
            #print('inicio',x)
            #print('fin',x_f)
            indices = np.where((new_freq >= x) & (new_freq <= x_f))[0]
            index_band_pass.append(indices)
            band_pass_prom.append(np.mean(Ti_total[i][indices]))

    band_pass_prom_t=np.reshape(band_pass_prom, (np.size(array1_tiempo),size_arr_freq))
    sub_band_def=np.reshape(sub_band_def, (np.size(array1_tiempo),size_arr_freq*2))
    return sub_band_def, Noise_total_time,Ti_total,band_pass_prom_t,index_band_pass

