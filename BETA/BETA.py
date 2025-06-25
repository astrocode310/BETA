import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from numpy import exp, pi, sqrt
from lmfit import Model
from scipy.optimize import curve_fit
import plotly.express as px
from matplotlib.ticker import MaxNLocator

class BETA:
    def __init__(self):
        self.search_result = None
        self.time = None
        self.flux = None
        self.error = None
        self.epoch = None

    def lc_download(self, alvo, missao="TESS", cadence=None, verbose=True):
        """
        Baixa a curva de luz do TESS para o alvo especificado.

        Parâmetros:
        - alvo: TIC ID, coordenadas ou nome do alvo.
        - missao: Missão espacial (padrão = "TESS").
        - cadence: 'short' ou 'long' (opcional).
        - verbose: Se True, imprime os setores encontrados.

        Retorna:
        - Um objeto search_result do lightkurve.
        """
        self.search_result = lk.search_lightcurve(alvo, mission=missao, cadence=cadence)

        if len(self.search_result) == 0:
            print("Nenhum resultado encontrado para o alvo especificado.")
            return None

        if verbose:
            print(f"Foram encontrados {len(self.search_result)} resultados:")
            print(self.search_result)

        return self.search_result

    def lc_processing(self, selected_lcs):
        """
        Processa a curva de luz para os setores e cadências desejados.
        Retorna o gráfico da curva de luz e salva os dados em um arquivo de texto.
        """
        if self.search_result is None:
            raise ValueError("Primeiro, baixe a curva de luz usando 'baixar_curva_luz'.")

        if not selected_lcs:
            raise ValueError("Nenhuma curva de luz selecionada para os setores e cadências fornecidos.")

        time0 = []
        flux0 = []
        err0 = []

        for lc_result in selected_lcs:
            lc = lc_result.download().normalize()
            if lc is not None:
                times = lc['time'].value
                flux_corr = lc['flux'].value
                err_corr = lc['flux_err'].value

                # Remove NaN values
                valid_indices = ~np.isnan(flux_corr)
                time0.append(times[valid_indices])
                flux0.append(flux_corr[valid_indices])
                err0.append(err_corr[valid_indices])


        # Gerar a curva de luz única
        self.time = np.array([item for sublist in time0 for item in sublist])
        self.flux = np.array([item for sublist in flux0 for item in sublist])
        self.error = np.array([item for sublist in err0 for item in sublist])

        # Plotar a curva de luz normalizada
        fig, ax = plt.subplots(figsize=(18, 5))
        ax.plot(self.time, self.flux, label='Dados', color='black')
        ax.set_xlabel('Time - 2457000 (BTJD dias)', size=16)
        ax.set_ylabel('Fluxo Normalizado', size=16)
        #ax.set_title('Curva de Luz Normalizada')
        ax.tick_params(labelsize=13)
        plt.tight_layout()
        plot_path = 'curva_luz_normalizada.pdf'
        plt.savefig(plot_path)
        plt.plot()
        
        figg = px.scatter(x=self.time, y=self.flux)
        figg.show()

        # Salvar em arquivo de texto
        txt_path = 'curva_luz_normalizada.txt'
        np.savetxt(txt_path, np.column_stack([self.time, self.flux, self.error]), header='Tempo\tFluxo Normalizado\tErro', fmt='%.6f')
        return plot_path, txt_path



    
    def calcular_epoca_inicial(self, t0, t1, frac_depth=0.06):
        """
        Calcula a época inicial ajustando uma gaussiana no mínimo primário.
        Retorna o plot e o valor da época inicial.
        """
        
        mask = (self.time >= t0) & (self.time <= t1)
        t_fit = self.time[mask]
        f_fit = self.flux[mask]
        
        #Calcular o tempo de trânsito
        flux_base = np.median(f_fit[t_fit < t_fit.mean() - 0.1])
        flux_min = np.min(f_fit)
        depth = flux_base - flux_min
        threshold = flux_base - frac_depth * depth

        in_transit = np.where(f_fit < threshold)[0]

        if len(in_transit) > 0:
            ingress_time = t_fit[in_transit[0]]
            egress_time = t_fit[in_transit[-1]]
        else:
            print('Ocorreu um erro durante o ajuste.')
        
            
        # Função gaussiana
        def gauss(x, A, mu, sigma, offset):
            return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + offset

        def ajustar_minimo_gaussiano(janela=0.5):
            idx_min = np.argmin(f_fit)
            t_central = t_fit[idx_min]

            # Estimativas iniciais: A (profundidade negativa), mu (posição), sigma, offset
            A0 = np.min(f_fit) - np.median(f_fit)
            mu0 = t_central
            sigma0 = (t_fit[-1] - t_fit[0]) / 6
            offset0 = np.median(f_fit)
            p0 = [A0, mu0, sigma0, offset0]

            popt, _ = curve_fit(gauss, t_fit, f_fit, p0=p0)
            
            self.epoch=popt[1]
            
            
            if True:
                plt.figure(figsize=(8, 4))
                plt.plot(t_fit, f_fit, 'ko', label='Dados')
                plt.plot(t_fit, gauss(t_fit, *popt), 'r-', label='Ajuste Gaussiano')
                plt.axvline(popt[1], color='blue', linestyle='--', label=f"T₀ = {popt[1]:.5f}")
                plt.axvline(ingress_time, color='black', linestyle='--', label=f'Início trânsito: {ingress_time:.2f}')
                plt.axvline(egress_time, color='black', linestyle='--', label=f'Final trânsito: {egress_time:.2f}')
                plt.xlabel("Tempo")
                plt.ylabel("Fluxo")
                plt.title("Ajuste Gaussiano ao Eclipse")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.show()
                
            return popt, ingress_time, egress_time
        return ajustar_minimo_gaussiano()
            

    def periodogram(self, harmonicos=4):
        """
        Calcula o periodograma, o valor do período e o diagrama de fase.
        Retorna o plot do periodograma, o valor do período e o diagrama de fase.
        """
        if self.time is None or self.flux is None:
            raise ValueError("Primeiro, processe a curva de luz usando \'processar_curva_luz\'.")

        # Calcular o periodograma
        lc1 = lk.LightCurve(time=self.time, flux=self.flux)
        
        pg = lc1.to_periodogram(oversample_factor=1)
        period = pg.period_at_max_power.value
        
        #Plotar periodograma
        pg.plot()
        plt.savefig('Periodograma', bbox_inches='tight')

        # Plotar o diagrama de fase
        lc1.fold(period=harmonicos*period).scatter()
        plt.savefig('Diagrama de fase', bbox_inches='tight')

        return f'O período calculado é de {period:.2f} dias'
    
    def get_period(self, min1, min2):
        """
        Calcula o período orbital através de ajustes gaussianos nos mínimos primários ou secundários.
        """
        ranges=[min1, min2]
        pars=[]
        
        def gauss(x, A, mu, sigma, offset):
            return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + offset

        def ajustar_minimo_gaussiano(janela=0.5):
            for rang in ranges:
                mask = (self.time >= rang[0]) & (self.time <= rang[1])
                t_fit = self.time[mask]
                f_fit = self.flux[mask]

                idx_min = np.argmin(f_fit)
                t_central = t_fit[idx_min]

                # Estimativas iniciais: A (profundidade negativa), mu (posição), sigma, offset
                A0 = np.min(f_fit) - np.median(f_fit)
                mu0 = t_central
                sigma0 = (t_fit[-1] - t_fit[0]) / 6
                offset0 = np.median(f_fit)
                p0 = [A0, mu0, sigma0, offset0]

                popt, _ = curve_fit(gauss, t_fit, f_fit, p0=p0)
                pars.append(popt)
                
        ajustar_minimo_gaussiano() #Chamar a função e calcular os parâmetros
      
        size_=23
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[15,5], constrained_layout=True)
        #fig.subplots_adjust(wspace=0.12)
        mask1 = (self.time >= min1[0]) & (self.time <= min1[1])
        ax1.set_title('Primário 1', size=size_)
        ax1.set_xlabel('Tempo (BTJD)', size=size_)
        ax1.set_ylabel('Fluxo Normalizado', size=size_)
        ax1.scatter(self.time[mask1], self.flux[mask1], s=12, color='black', label='Dados')
        ax1.plot(self.time[mask1], gauss(self.time[mask1], pars[0][0], pars[0][1], pars[0][2], pars[0][3]),color='r', label='Ajuste gaussiano')
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax1.tick_params(labelsize=20)
        ax1.axvline(x=pars[0][1], linestyle='--', label='Mínimo Calculado', color='b')
        ax1.axvline(x=pars[0][1], linestyle='--', label='Época inicial', color='b')
        ax1.legend(loc='lower left', fontsize=13)
        
        
        mask2 = (self.time >= min2[0]) & (self.time <= min2[1])
        ax2.set_title('Primário 2', size=size_)
        ax2.set_xlabel('Tempo (BTJD)', size=size_)
        #ax2.set_ylabel('Fluxo Normalizado', size=size_)
        ax2.scatter(self.time[mask2], self.flux[mask2], s=12, color='black',label='Dados')
        ax2.plot(self.time[mask2], gauss(self.time[mask2], pars[1][0], pars[1][1], pars[1][2], pars[1][3]),color='r', label='Ajuste gaussiano')
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax2.tick_params(labelsize=20)
        ax2.set_yticklabels([])
        ax2.axvline(x=pars[1][1], linestyle='--', label='Mínimo calculado', color='b')
        ax2.legend(loc='lower left', fontsize=13)
        
        """
        mask3 = (self.time >= min3[0]) & (self.time <= min3[1])
        ax3.set_title('Primário 3', size=size_)
        #ax3.set_xlabel('Time - 2457000 (BTJD dias)', size=size_)
        #ax3.set_ylabel('Fluxo Normalizado', size=size_)
        ax3.scatter(self.time[mask3], self.flux[mask3], s=12, color='black',label='Dados')
        ax3.plot(self.time[mask3], gauss(self.time[mask3], pars[2][0], pars[2][1], pars[2][2], pars[2][3]),color='r', label='Ajuste gaussiano')
        ax3.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax3.tick_params(labelsize=13)
        ax3.set_yticklabels([])
        ax3.axvline(x=pars[2][1], linestyle='--', label='Mínimo calculado')
        ax3.legend(loc='lower left', fontsize=11)
        """
        plt.savefig('Ajuste de gaussianas.jpg', bbox_inches='tight')
        plt.show()
        
        period_measured=(pars[1][1]-pars[0][1])
        
        #Calcular e plotar a fase
        #if self.epoch is not None:
        phase=(self.time-self.epoch)/period_measured % 1
        
        """
        plt.figure(figsize=[18,5])
        plt.scatter(phase, self.flux, color='black', s=12)
        plt.xlabel('Fase', size=size_)
        plt.ylabel('Fluxo', size=size_)
        plt.tick_params(labelsize=20)
        plt.savefig('Diagrama de fase.jpg', bbox_inches='tight')
        plt.show() 
       # else:
        #    print('A época inicial deve ser calculada')
        """
        # Duplicar a curva com deslocamento
        phase_extended = np.concatenate([phase - 1, phase, phase + 1])
        flux_extended = np.concatenate([self.flux, self.flux, self.flux])

        # Filtrar apenas o intervalo desejado
        mask = (phase_extended >= -0.25) & (phase_extended <= 1.25)

        plt.figure(figsize=(18, 5))
        plt.scatter(phase_extended[mask], flux_extended[mask], color='black', s=12)
        #plt.plot(phase_extended[mask], flux_extended[mask], '.', markersize=2, alpha=0.6)
        plt.xlabel('Fase Orbital', size=size_)
        plt.ylabel('Fluxo', size=size_)
        plt.tick_params(labelsize=20)
        plt.xticks([-0.25, 0.00, 0.25, 0.50, 0.75, 1.00, 1.25])
        plt.savefig('Diagrama de fase.jpg', bbox_inches='tight')
        plt.show()
        
       
        return pars, period_measured, phase
    





