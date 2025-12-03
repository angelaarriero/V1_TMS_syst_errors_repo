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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

def graficas_paper_TMS(total_BEM_atmos_cmb,total_I1_BEM_atmos_cmb,total_I2_BEM_atmos_cmb,freq_spec,T_sky_bet_total,T_load_bet_total,Tlss_1,Tlss_2,Tlss_3,Tlss_4,Gtot,Tn_tot_sky,Tn_tot_load):
    ###########............................................................########################################
    ###########............................................................########################################
    ###########............................................................########################################
    BEM_plot=total_BEM_atmos_cmb[100]
    new_freq=freq_spec[0]*1e9
    band_pass_prom_BEM=[]
    size_arr_freq=int((new_freq[np.size(new_freq)-1]-new_freq[0])/250e6)
    index_band_pass=[]
    sub_band_def=[]

    for j in range(size_arr_freq):
        x = new_freq[0]+(240e6*j)+(j*1e7)
        x_f = x+240e6
        sub_band_def.append(np.array([x,x_f]))
        #print('inicio',x)
        #print('fin',x_f)
        indices = np.where((new_freq >= x) & (new_freq <= x_f))[0]
        index_band_pass.append(indices)
        band_pass_prom_BEM.append(np.mean(BEM_plot[indices]))

    mean_BEM=np.mean(BEM_plot)
    
    mean_bskyA1=np.mean(T_sky_bet_total)
    mean_bloadA1=np.mean(T_load_bet_total)
    desv_Tsky=np.std(T_sky_bet_total)
    desv_Tload=np.std(T_load_bet_total)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(freq_spec[0],T_sky_bet_total,color='skyblue',linestyle='solid')
    plt.axhline(mean_bskyA1, color = 'skyblue', linewidth = 1,label=r'$\langle \beta_{\rm eff}^{\rm sky} \rangle$ = %.2f $\pm$ %.3f' % (mean_bskyA1, desv_Tsky),linestyle='--')
    ax1.plot(freq_spec[0],T_load_bet_total,color='coral',linestyle='solid')
    plt.axhline(mean_bloadA1, color = 'coral', linewidth = 1,label=r'$ \langle \beta_{\rm eff}^{\rm load} \rangle$ = %.2f $\pm$ %.3f' % (mean_bloadA1,desv_Tload),linestyle='--')
    ax1.tick_params(labelsize=15,axis='y',which='both')
    ax1.set_ylabel(r'$\rm \beta_{eff}$',fontsize=20)
    ax1.set_xlabel(r'$\nu \ [GHz]$',fontsize=20)
    ax1.legend(loc='best', fontsize=11)
    ax1.tick_params(labelsize=15,axis='both',which='both')
    plt.savefig('/home/aarriero/Documents/Angela_cmb/four_year/cap2_TOD_TMS/TOD_V_dic2025/B_results_plots/B_betas_offset_t.pdf')
    
    ###########............................................................########################################
    ###########............................................................########################################
    ###########............................................................########################################
    Tlss_1 = np.array(Tlss_1)
    Tlss_2 = np.array(Tlss_2)
    Tlss_3 = np.array(Tlss_3)
    Tlss_4 = np.array(Tlss_4)

    Tloss_sky=(Tlss_1+Tlss_2+Tlss_3+Tlss_4)
    Tloss_load=(Tlss_1-Tlss_2+Tlss_3-Tlss_4)

    resta_loss=Tloss_sky-Tloss_load
    resta_loss_mean=np.mean(resta_loss[1]/Gtot)
    resta_loss_desv=np.std(resta_loss[1]/Gtot)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(freq_spec[0],resta_loss[1]/Gtot,linestyle='-',color='plum')
    plt.axhline(resta_loss_mean, color = "plum", linewidth = 1,label=r'$ \langle \, T^{off}_{eff} \, \rangle $ = %.2f $\pm$ %.3f' % (resta_loss_mean, resta_loss_desv),linestyle='--')
    plt.xlabel(r'$\nu \ [GHz]$',fontsize=20)
    plt.ylabel(r'$ \rm T^{off}_{eff} \ [K]$', fontsize=20)
    plt.legend(loc='best', fontsize=11)
    plt.tick_params(labelsize=15,axis='both',which='both')
    plt.savefig('/home/aarriero/Documents/Angela_cmb/four_year/cap2_TOD_TMS/TOD_V_dic2025/B_results_plots/B_LOSS_T_bem_DC_exc.pdf')
    
    
    ###########............................................................########################################
    ###########............................................................########################################
    ###########............................................................########################################
    min_arr=((freq_spec[0][1]-freq_spec[0][0])/2)+freq_spec[0][0]
    max_arr=freq_spec[0][np.size(freq_spec[0])-1]-((freq_spec[0][1]-freq_spec[0][0])/2)
    arra_freq_new=np.linspace(min_arr,max_arr,40)
    x = arra_freq_new 
    y2= band_pass_prom_BEM-(mean_BEM*np.ones(np.size(band_pass_prom_BEM)))
    #error_horizontal = ((arr22[1][1]-arr22[1][0])/1e9 )+0.01
    error_horizontal =0.125
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # ========================== PLOT 1 ===============================
    axs[0].plot(
        freq_spec[0], 
        BEM_plot,
        color='hotpink',
        label=r'$\rm \langle \Delta T \rangle_{\rm TMS} = %.2f \pm %.3f$' % (np.mean(BEM_plot), np.std(BEM_plot)))

    axs[0].set_xlabel(r'$\rm \nu\;[\rm GHz]$', fontsize=20)
    axs[0].set_ylabel(r'$\rm \Delta T\, [\rm K]$', fontsize=20)
    axs[0].tick_params(labelsize=15)
    axs[0].legend(loc='best', fontsize=12)
    #axs[0].grid(alpha=0.3)

    # ========================== PLOT 2 ===============================
    axs[1].errorbar(x, y2, xerr=error_horizontal, fmt='o',color='hotpink',ecolor='k',capsize=1)

    axs[1].set_xlabel(r'$\rm \nu\;[\rm GHz]$', fontsize=20)
    axs[1].set_ylabel(r'$\rm \Delta T^a_{\rm sub}\;[\rm K]$', fontsize=20)
    axs[1].tick_params(labelsize=15)
    #axs[1].legend(loc='best', fontsize=10)
    #axs[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/aarriero/Documents/Angela_cmb/four_year/cap2_TOD_TMS/TOD_V_dic2025/B_results_plots/B_abs_resul_total.pdf')
    
    ###########............................................................########################################
    ###########............................................................########################################
    ###########............................................................########################################
    # =============================
    # 1. Estadísticos
    # =============================
    Tn_tot_sky_mean  = np.mean(Tn_tot_sky)
    Tn_tot_sky_std   = np.std(Tn_tot_sky)

    Tn_tot_load_mean = np.mean(Tn_tot_load)
    Tn_tot_load_std  = np.std(Tn_tot_load)

    total = Tn_tot_sky - Tn_tot_load
    total_mean = np.mean(total)
    total_std  = np.std(total)

    # =============================
    # 2. Gráfica
    # =============================
    plt.figure(figsize=(10, 5))

    # Curvas principales
    plt.plot(freq_spec[0], Tn_tot_sky,
             linestyle='-',  color='skyblue')

    plt.plot(freq_spec[0], Tn_tot_load, 
             linestyle='-', color='coral', linewidth=1)

    plt.plot(freq_spec[0], total, 
             linestyle='-', color='g', linewidth=1)

    # Líneas horizontales de las medias
    plt.axhline(
        Tn_tot_sky_mean, 
        color="skyblue", linestyle='--', linewidth=1,
        label=r'$\rm \langle T^{noise}_{sky} \rangle = %.2f \pm %.3f$' 
              % (Tn_tot_sky_mean, Tn_tot_sky_std)
    )

    plt.axhline(
        Tn_tot_load_mean,
        color="coral", linestyle='--', linewidth=1,
        label=r'$\rm \langle T^{noise}_{load} \rangle = %.2f \pm %.3f$' 
              % (Tn_tot_load_mean, Tn_tot_load_std)
    )

    plt.axhline(
        total_mean,
        color="g", linestyle='--', linewidth=1,
        label=r'$\rm \langle T^{noise}_{eff} \rangle = %.2f \pm %.3f$' 
              % (total_mean, total_std)
    )

    # =============================
    # 3. Formato
    # =============================
    plt.xlabel(r'$\rm \nu \ [GHz]$', fontsize=20)
    plt.ylabel(r'$\rm T^{eff}_{noise}\ [K]$', fontsize=20)

    plt.tick_params(labelsize=15)
    plt.legend(loc='best', fontsize=10, ncol=3)

    plt.tight_layout()

    plt.savefig('/home/aarriero/Documents/Angela_cmb/four_year/cap2_TOD_TMS/TOD_V_dic2025/B_results_plots/B_Tnoise_gain_noise.pdf')

    return 0