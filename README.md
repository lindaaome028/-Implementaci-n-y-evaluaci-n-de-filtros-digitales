# -Implementaci-n-y-evaluaci-n-de-filtros-digitales
proyecto escolar
"""
Implementación y evaluación de filtros digitales
Autor: [Linda ROMERO]
Fecha: 2026
Descripción: Diseño e implementación de filtros pasa bajos, pasa altos y pasa bandas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

# Configuración de parámetros
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

class FiltrosDigitales:
    def __init__(self, fs=1000):
        """
        Inicializa la clase con la frecuencia de muestreo
        
        Parámetros:
        fs: Frecuencia de muestreo en Hz
        """
        self.fs = fs
        self.nyquist = fs / 2
        
    def generar_senal_prueba(self, duracion=2):
        """
        Genera una señal de prueba compuesta y ruido
        
        Parámetros:
        duracion: Duración de la señal en segundos
        
        Retorna:
        t: Vector de tiempo
        senal_original: Señal limpia
        senal_ruidosa: Señal con ruido
        """
        # Vector de tiempo
        t = np.linspace(0, duracion, int(self.fs * duracion))
        
        # Componentes de frecuencia (en Hz)
        f1, f2, f3 = 50, 150, 300
        
        # Señal limpia compuesta
        senal_original = (np.sin(2 * np.pi * f1 * t) + 
                         0.5 * np.sin(2 * np.pi * f2 * t) + 
                         0.3 * np.sin(2 * np.pi * f3 * t))
        
        # Añadir ruido blanco
        ruido = 0.5 * np.random.randn(len(t))
        senal_ruidosa = senal_original + ruido
        
        return t, senal_original, senal_ruidosa
    
    def calcular_espectro(self, senal):
        """
        Calcula el espectro de frecuencia de una señal
        
        Parámetros:
        senal: Señal de entrada
        
        Retorna:
        frecuencias: Vector de frecuencias
        magnitud: Magnitud del espectro
        """
        n = len(senal)
        frecuencias = fftfreq(n, 1/self.fs)[:n//2]
        transformada = fft(senal)
        magnitud = np.abs(transformada[:n//2])
        return frecuencias, magnitud
    
    def disenar_filtro_pasabajos(self, corte, orden=4, tipo='butter'):
        """
        Diseña un filtro pasa bajos
        
        Parámetros:
        corte: Frecuencia de corte en Hz
        orden: Orden del filtro
        tipo: Tipo de filtro ('butter', 'cheby1', 'cheby2')
        """
        corte_normalizado = corte / self.nyquist
        
        if tipo == 'butter':
            b, a = signal.butter(orden, corte_normalizado, btype='low')
        elif tipo == 'cheby1':
            b, a = signal.cheby1(orden, 0.5, corte_normalizado, btype='low')
        elif tipo == 'cheby2':
            b, a = signal.cheby2(orden, 40, corte_normalizado, btype='low')
        
        return b, a
    
    def disenar_filtro_pasaaltos(self, corte, orden=4, tipo='butter'):
        """
        Diseña un filtro pasa altos
        """
        corte_normalizado = corte / self.nyquist
        
        if tipo == 'butter':
            b, a = signal.butter(orden, corte_normalizado, btype='high')
        elif tipo == 'cheby1':
            b, a = signal.cheby1(orden, 0.5, corte_normalizado, btype='high')
        
        return b, a
    
    def disenar_filtro_pasabanda(self, corte_bajo, corte_alto, orden=4):
        """
        Diseña un filtro pasa banda Butterworth
        """
        corte_normalizado = [corte_bajo / self.nyquist, corte_alto / self.nyquist]
        b, a = signal.butter(orden, corte_normalizado, btype='band')
        return b, a
    
    def disenar_fir_ventana(self, corte, ancho_transicion, btype='low'):
        """
        Diseña un filtro FIR usando método de ventana
        """
        # Calcular orden del filtro usando la regla de Kaiser
        orden = int(4 * self.fs / ancho_transicion)
        if orden % 2 == 0:
            orden += 1
            
        # Frecuencia de corte normalizada
        corte_normalizado = corte / self.nyquist
        
        # Diseñar filtro FIR con ventana de Hamming
        if btype == 'low':
            b = signal.firwin(orden, corte_normalizado, window='hamming')
            a = [1.0]
        elif btype == 'high':
            b = signal.firwin(orden, corte_normalizado, window='hamming', pass_zero=False)
            a = [1.0]
        
        return b, a
    
    def graficar_resultados(self, t, senal_original, senal_ruidosa, senal_filtrada, 
                           titulo="Resultados del Filtrado"):
        """
        Grafica las señales en tiempo y frecuencia
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Señales en el tiempo
        axes[0, 0].plot(t[:500], senal_original[:500], 'b-', label='Original', alpha=0.7)
        axes[0, 0].plot(t[:500], senal_ruidosa[:500], 'r-', label='Ruidosa', alpha=0.5)
        axes[0, 0].set_xlabel('Tiempo (s)')
        axes[0, 0].set_ylabel('Amplitud')
        axes[0, 0].set_title('Señales en el tiempo (antes del filtrado)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(t[:500], senal_original[:500], 'b-', label='Original', alpha=0.7)
        axes[0, 1].plot(t[:500], senal_filtrada[:500], 'g-', label='Filtrada', alpha=0.7)
        axes[0, 1].set_xlabel('Tiempo (s)')
        axes[0, 1].set_ylabel('Amplitud')
        axes[0, 1].set_title('Señales en el tiempo (después del filtrado)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Espectros de frecuencia
        frec, mag_original = self.calcular_espectro(senal_original)
        frec, mag_ruidosa = self.calcular_espectro(senal_ruidosa)
        frec, mag_filtrada = self.calcular_espectro(senal_filtrada)
        
        axes[1, 0].plot(frec, mag_original, 'b-', label='Original', alpha=0.7)
        axes[1, 0].plot(frec, mag_ruidosa, 'r-', label='Ruidosa', alpha=0.5)
        axes[1, 0].set_xlabel('Frecuencia (Hz)')
        axes[1, 0].set_ylabel('Magnitud')
        axes[1, 0].set_title('Espectros de frecuencia (antes del filtrado)')
        axes[1, 0].set_xlim([0, 500])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(frec, mag_original, 'b-', label='Original', alpha=0.7)
        axes[1, 1].plot(frec, mag_filtrada, 'g-', label='Filtrada', alpha=0.7)
        axes[1, 1].set_xlabel('Frecuencia (Hz)')
        axes[1, 1].set_ylabel('Magnitud')
        axes[1, 1].set_title('Espectros de frecuencia (después del filtrado)')
        axes[1, 1].set_xlim([0, 500])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(titulo, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def graficar_respuesta_frecuencia(self, b, a, titulo="Respuesta en Frecuencia"):
        """
        Grafica la respuesta en frecuencia del filtro
        """
        w, h = signal.freqz(b, a, worN=8000)
        frec = w * self.fs / (2 * np.pi)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Magnitud
        ax1.plot(frec, 20 * np.log10(np.abs(h)), 'b-', linewidth=2)
        ax1.set_xlabel('Frecuencia (Hz)')
        ax1.set_ylabel('Magnitud (dB)')
        ax1.set_title(f'{titulo} - Magnitud')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 500])
        
        # Fase
        ax2.plot(frec, np.angle(h), 'r-', linewidth=2)
        ax2.set_xlabel('Frecuencia (Hz)')
        ax2.set_ylabel('Fase (radianes)')
        ax2.set_title(f'{titulo} - Fase')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 500])
        
        plt.tight_layout()
        plt.show()


# ==========================
# PROGRAMA PRINCIPAL
# ==========================

def main():
    print("=" * 60)
    print("IMPLEMENTACIÓN Y EVALUACIÓN DE FILTROS DIGITALES")
    print("=" * 60)
    
    # Crear instancia del procesador de filtros
    procesador = FiltrosDigitales(fs=1000)
    
    # 1. Generar señal de prueba
    print("\n1. Generando señal de prueba...")
    t, senal_original, senal_ruidosa = procesador.generar_senal_prueba(duracion=2)
    
    # 2. FILTRO PASA BAJOS
    print("\n2. Aplicando FILTRO PASA BAJOS (corte = 100 Hz)")
    b_lp, a_lp = procesador.disenar_filtro_pasabajos(corte=100, orden=5)
    senal_filtrada_lp = signal.filtfilt(b_lp, a_lp, senal_ruidosa)
    
    # Graficar respuesta del filtro pasa bajos
    procesador.graficar_respuesta_frecuencia(b_lp, a_lp, "Filtro Pasa Bajos - Butterworth")
    
    # Graficar resultados del filtrado pasa bajos
    procesador.graficar_resultados(t, senal_original, senal_ruidosa, senal_filtrada_lp,
                                  "FILTRO PASA BAJOS - fc = 100 Hz")
    
    # 3. FILTRO PASA ALTOS
    print("\n3. Aplicando FILTRO PASA ALTOS (corte = 200 Hz)")
    b_hp, a_hp = procesador.disenar_filtro_pasaaltos(corte=200, orden=5)
    senal_filtrada_hp = signal.filtfilt(b_hp, a_hp, senal_ruidosa)
    
    # Graficar respuesta del filtro pasa altos
    procesador.graficar_respuesta_frecuencia(b_hp, a_hp, "Filtro Pasa Altos - Butterworth")
    
    # Graficar resultados del filtrado pasa altos
    procesador.graficar_resultados(t, senal_original, senal_ruidosa, senal_filtrada_hp,
                                  "FILTRO PASA ALTOS - fc = 200 Hz")
    
    # 4. FILTRO PASA BANDA
    print("\n4. Aplicando FILTRO PASA BANDA (80 Hz - 200 Hz)")
    b_bp, a_bp = procesador.disenar_filtro_pasabanda(corte_bajo=80, corte_alto=200, orden=5)
    senal_filtrada_bp = signal.filtfilt(b_bp, a_bp, senal_ruidosa)
    
    # Graficar respuesta del filtro pasa banda
    procesador.graficar_respuesta_frecuencia(b_bp, a_bp, "Filtro Pasa Banda - Butterworth")
    
    # Graficar resultados del filtrado pasa banda
    procesador.graficar_resultados(t, senal_original, senal_ruidosa, senal_filtrada_bp,
                                  "FILTRO PASA BANDA - 80 Hz a 200 Hz")
    
    # 5. COMPARACIÓN DE FILTROS FIR vs IIR
    print("\n5. Comparando filtros FIR vs IIR (Pasa Bajos 100 Hz)")
    
    # Filtro FIR
    b_fir, a_fir = procesador.disenar_fir_ventana(corte=100, ancho_transicion=50, btype='low')
    senal_filtrada_fir = signal.filtfilt(b_fir, a_fir, senal_ruidosa)
    
    # Filtro IIR (Butterworth)
    b_iir, a_iir = procesador.disenar_filtro_pasabajos(corte=100, orden=5, tipo='butter')
    senal_filtrada_iir = signal.filtfilt(b_iir, a_iir, senal_ruidosa)
    
    # Graficar comparación
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Señales filtradas
    axes[0, 0].plot(t[:500], senal_original[:500], 'b-', label='Original', alpha=0.7)
    axes[0, 0].plot(t[:500], senal_filtrada_fir[:500], 'r-', label='FIR', alpha=0.7)
    axes[0, 0].set_title('Comparación - FIR')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t[:500], senal_original[:500], 'b-', label='Original', alpha=0.7)
    axes[0, 1].plot(t[:500], senal_filtrada_iir[:500], 'g-', label='IIR', alpha=0.7)
    axes[0, 1].set_title('Comparación - IIR')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Respuestas en frecuencia
    w_fir, h_fir = signal.freqz(b_fir, a_fir, worN=8000)
    w_iir, h_iir = signal.freqz(b_iir, a_iir, worN=8000)
    frec = w_fir * procesador.fs / (2 * np.pi)
    
    axes[1, 0].plot(frec, 20 * np.log10(np.abs(h_fir)), 'r-', label='FIR', linewidth=2)
    axes[1, 0].set_title('Respuesta FIR')
    axes[1, 0].set_xlabel('Frecuencia (Hz)')
    axes[1, 0].set_ylabel('Magnitud (dB)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, 500])
    
    axes[1, 1].plot(frec, 20 * np.log10(np.abs(h_iir)), 'g-', label='IIR', linewidth=2)
    axes[1, 1].set_title('Respuesta IIR')
    axes[1, 1].set_xlabel('Frecuencia (Hz)')
    axes[1, 1].set_ylabel('Magnitud (dB)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 500])
    
    plt.suptitle("COMPARACIÓN FILTROS FIR vs IIR", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 6. ANÁLISIS DE EFECTIVIDAD
    print("\n" + "=" * 60)
    print("ANÁLISIS DE EFECTIVIDAD DE FILTROS")
    print("=" * 60)
    
    # Calcular SNR antes y después del filtrado
    def calcular_snr(senal_limpia, senal_con_ruido):
        potencia_senal = np.mean(senal_limpia**2)
        potencia_ruido = np.mean((senal_con_ruido - senal_limpia)**2)
        if potencia_ruido > 0:
            return 10 * np.log10(potencia_senal / potencia_ruido)
        return float('inf')
    
    # SNR original
    snr_original = calcular_snr(senal_original, senal_ruidosa)
    
    # SNR después de cada filtro
    snr_lp = calcular_snr(senal_original, senal_filtrada_lp)
    snr_hp = calcular_snr(senal_original, senal_filtrada_hp)
    snr_bp = calcular_snr(senal_original, senal_filtrada_bp)
    snr_fir = calcular_snr(senal_original, senal_filtrada_fir)
    snr_iir = calcular_snr(senal_original, senal_filtrada_iir)
    
    print(f"\nRelación Señal-Ruido (SNR) en dB:")
    print(f"Señal original con ruido: {snr_original:.2f} dB")
    print(f"Después de filtro pasa bajos: {snr_lp:.2f} dB (Mejora: {snr_lp - snr_original:.2f} dB)")
    print(f"Después de filtro pasa altos: {snr_hp:.2f} dB (Mejora: {snr_hp - snr_original:.2f} dB)")
    print(f"Después de filtro pasa banda: {snr_bp:.2f} dB (Mejora: {snr_bp - snr_original:.2f} dB)")
    print(f"Después de filtro FIR: {snr_fir:.2f} dB (Mejora: {snr_fir - snr_original:.2f} dB)")
    print(f"Después de filtro IIR: {snr_iir:.2f} dB (Mejora: {snr_iir - snr_original:.2f} dB)")
    
    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 60)

if __name__ == "__main__":
    main()
