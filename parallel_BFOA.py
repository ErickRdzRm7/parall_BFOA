import copy # Importa el módulo completo. Ahora puedes usar copy.deepcopy()
from multiprocessing import Manager, Pool, Value # Value es necesario para globalNFE
import time
from bacteria import bacteria # Asegúrate que importe la clase modificada abajo
import numpy as np # Es convención usar np para numpy
import pandas as pd
import matplotlib.pyplot as plt
from fastaReader import fastaReader
# Asegúrate que el archivo 'bacteria.py' esté en el mismo directorio o en el PYTHONPATH
# y que contenga la clase 'bacteria' con las modificaciones anteriores (caché, NFE corregido, etc.)

if __name__ == "__main__":
    # --- PARÁMETROS PRINCIPALES ---
    numeroDeBacterias = 10
    iteraciones = 100  # <--- 1. AUMENTADO SIGNIFICATIVAMENTE
    tumbo_gaps = 2     # Número de gaps a insertar en cada tumbo
    # nado_steps = 4   # <--- Parámetro para futura implementación de Swim (Ns)

    # --- Parámetros de Interacción BFOA ---
    # ¡¡¡ REQUIEREN AJUSTE FINO DESPUÉS DE VER RESULTADOS !!!
    # Estos son solo puntos de partida sugeridos para aumentar un poco la influencia.
    dAttr = 0.5       # Magnitud de atracción (Mantener por ahora)
    wAttr = 0.0001    # <--- 2. AJUSTADO (Reducido para decaimiento más lento / mayor alcance)
    hRep = 0.5        # Magnitud de repulsión (Mantener por ahora, 'd' en formula)
    wRepel = 0.00005  # <--- 2. AJUSTADO (Reducido para decaimiento más lento / mayor alcance)
    # ------------------------------------

    secuencias = list()
    resultados = []
    
    try:
        reader = fastaReader() # Instanciar una vez
        secuencias = reader.seqs
        names = reader.names # Guardar nombres para salida FASTA
        if not secuencias:
            print("Error: No se cargaron secuencias desde fastaReader.")
            exit() 
    except Exception as e:
        print(f"Error al leer el archivo FASTA: {e}")
        exit()
        
    # Hace todas las secuencias listas de caracteres
    for i in range(len(secuencias)):
        secuencias[i] = list(secuencias[i])

    # --- Inicialización de Manager y variables compartidas ---
    try:
        manager = Manager()
        globalNFE = manager.Value('i', 0) 
        poblacion = manager.list([None] * numeroDeBacterias) 
        fitness_cache = manager.dict() 
    except Exception as e:
        print(f"Error al inicializar multiprocessing.Manager: {e}")
        exit()

    numSec = len(secuencias)
    print(f"Número de secuencias: {numSec}")
    if numSec == 0:
        print("Error: El número de secuencias es 0.")
        exit()

    # --- Funciones Auxiliares ---
    def poblacionInicial(): 
        try:
            base_bacterium = tuple(tuple(s) for s in secuencias) 
            for i in range(numeroDeBacterias):
                poblacion[i] = copy.deepcopy(base_bacterium) 
        except Exception as e:
            print(f"Error durante la inicialización de la población: {e}")
            exit()

    def printPoblacion(pop, num_bacterias_a_imprimir=3):
        print(f"--- Mostrando primeras {num_bacterias_a_imprimir} bacterias ---")
        # ... (código de printPoblacion sin cambios) ...
        print("-------------------------------------")

    #---------------------------------------------------------------------------------------------------------
    # --- Instanciación del Operador Bacterial ---
    try:
        operadorBacterial = bacteria(numeroDeBacterias, fitness_cache, globalNFE)    
    except Exception as e:
        print(f"Error al instanciar la clase 'bacteria': {e}")
        exit()
    
    # --- Inicio del Algoritmo ---
    start_time = time.time()
    
    print("Creando población inicial...")
    poblacionInicial() 

    print("Cuadrando población inicial...")
    try:
        current_poblacion_list = list(poblacion) 
        operadorBacterial.cuadra(numSec, current_poblacion_list) 
        poblacion[:] = current_poblacion_list 
        print("Población inicial cuadrada.")
    except Exception as e:
        print(f"Error durante el cuadramiento inicial: {e}")
        exit()
    
    # --- Evaluación Inicial ---
    try:
        print("Evaluando Blosum inicial (con caché)...")
        operadorBacterial.evaluaBlosumInicial(poblacion) 
        print("Creando Tablas Atract/Repel iniciales...")
        # Pasar los parámetros correctos (hRep para magnitud de repulsión)
        operadorBacterial.creaTablasAtractRepel(poblacion, dAttr, wAttr, hRep, wRepel) 
        operadorBacterial.creaTablaInteraction()
        print("Creando Tabla Fitness inicial...")
        operadorBacterial.creaTablaFitness()
        print("Evaluación inicial completa.")
        initial_nfe = globalNFE.value
        bestIdx, bestFitness = operadorBacterial.obtieneBest(initial_nfe)
    except Exception as e:
        print(f"Error durante la evaluación inicial: {e}")
        exit()

    # --- Almacenamiento de Progreso ---
    mejores_fitness_iter = [bestFitness] 
    nfe_acumulado_iter = [initial_nfe] 

    # --- Bucle Principal de Iteraciones ---
    print(f"\n--- Iniciando {iteraciones} Iteraciones del Algoritmo ---")
    for it in range(iteraciones):
        iter_start_time = time.time()
        # Imprimir progreso cada cierto número de iteraciones para no saturar
        if (it + 1) % 10 == 0 or it == 0:
             print(f"\n--- Iteración {it+1}/{iteraciones} ---")
        else:
             print(f".", end='', flush=True) # Imprimir punto para indicar progreso

        try:
            # --- Quimiotaxis (Tumbo + Cuadra + Evaluación) ---
            
            # Aplicando Tumbo... (No imprimir en cada iteración si son muchas)
            current_poblacion_list = list(poblacion) 
            operadorBacterial.tumbo(numSec, current_poblacion_list, tumbo_gaps) 
            poblacion[:] = current_poblacion_list 

            # Cuadrando Alineamientos post-Tumbo...
            current_poblacion_list = list(poblacion)
            operadorBacterial.cuadra(numSec, current_poblacion_list)
            poblacion[:] = current_poblacion_list

            # Evaluando Blosum post-Tumbo (con caché)...
            operadorBacterial.evaluaBlosumConCache(poblacion) 

            # Creando Tablas Atract/Repel...
            operadorBacterial.creaTablasAtractRepel(poblacion, dAttr, wAttr, hRep, wRepel)
            operadorBacterial.creaTablaInteraction()
            # Creando Tabla Fitness...
            operadorBacterial.creaTablaFitness()
            # Evaluación de Fitness completada. (No imprimir en cada iteración)

            # --- Fin Quimiotaxis ---

            # --- Obtener Mejor y Mostrar Progreso ---
            current_nfe = globalNFE.value 
            bestIdx, bestFitness = operadorBacterial.obtieneBest(current_nfe) # Imprime el mejor actual
            
            # --- Almacenar resultados de ESTA iteración ---
            mejores_fitness_iter.append(bestFitness) 
            nfe_acumulado_iter.append(current_nfe) 
            
            # Imprimir resumen solo cada 10 iteraciones
            if (it + 1) % 10 == 0:
                 print(f"Iteración {it+1} completada. Mejor Fitness: {bestFitness:.4f}. NFE acumulado: {current_nfe}")
            
            # --- Selección/Reproducción Simple ---
            # Aplicando Selección (replaceWorst)... (No imprimir en cada iteración)
            current_poblacion_list = list(poblacion)
            operadorBacterial.replaceWorst(current_poblacion_list, bestIdx) # Imprime si reemplaza
            poblacion[:] = current_poblacion_list
            
            # Imprimir tiempo solo cada 10 iteraciones
            if (it + 1) % 10 == 0:
                 print(f"Tiempo acumulado parcial: {time.time() - start_time:.2f} segundos")

        except Exception as e:
            print(f"\nError durante la iteración {it+1}: {e}")
            print("Continuando con la siguiente iteración si es posible...")


    # --- Fin del Bucle Principal ---
    print("\nBucle de iteraciones finalizado.") # Añadir mensaje claro
    total_time = time.time() - start_time
    print(f"\n--- Optimización Finalizada ---")
    print(f"Tiempo Total: {total_time:.2f} segundos")
    try:
        final_nfe = globalNFE.value
        # Obtener el mejor final una última vez
        final_best_idx, final_best_fitness = operadorBacterial.obtieneBest(final_nfe)
        print(f"Mejor Fitness Final Encontrado: {final_best_fitness:.4f} (En Bacteria Índice {final_best_idx})")
        print(f"Total NFE (Evaluaciones Blosum únicas): {final_nfe}")
    except Exception as e:
        print(f"Error al obtener resultados finales: {e}")
        final_best_idx = -1 # Indicar que no se pudo obtener el índice final

    # --- Resultados y Gráfica ---
    num_resultados = len(mejores_fitness_iter)
    iteraciones_eje_x = list(range(num_resultados)) 

    df_resultados = pd.DataFrame({
        'Paso': iteraciones_eje_x, 
        'Mejor Fitness': mejores_fitness_iter,
        'NFE Acumulado': nfe_acumulado_iter 
    })

    print("\nResumen del Mejor Fitness y NFE por Paso (0=Inicial):")
    print(df_resultados.tail()) # Imprimir solo las últimas filas si son muchas

    # (Código para graficar y guardar CSV sin cambios)
    # ... (Copiar desde la versión anterior) ...
    try:
        plt.figure(figsize=(12, 6))
        
        ax1 = plt.subplot(1, 2, 1) 
        ax1.plot(df_resultados['Paso'], df_resultados['Mejor Fitness'], marker='o', linestyle='-', color='b', markersize=3) # Puntos más pequeños
        ax1.set_title(f"Progreso del Mejor Fitness\nNFE Total: {final_nfe}")
        ax1.set_xlabel("Paso (0 = Inicial)")
        ax1.set_ylabel("Mejor Fitness")
        # Escala logarítmica puede no ser necesaria ahora, pero la dejamos condicional
        if any(f > 0 for f in mejores_fitness_iter[1:]):
            try:
                 ax1.set_yscale('log')
            except ValueError:
                 print("Advertencia: No se pudo aplicar escala logarítmica al fitness.")
        ax1.grid(True)
        
        ax2 = plt.subplot(1, 2, 2) 
        ax2.plot(df_resultados['Paso'], df_resultados['NFE Acumulado'], marker='s', linestyle='--', color='r', markersize=3) # Puntos más pequeños
        ax2.set_title(f"NFE Acumulado por Paso\nNFE Total: {final_nfe}")
        ax2.set_xlabel("Paso (0 = Inicial)")
        ax2.set_ylabel("NFE Acumulado")
        ax2.grid(True)

        plt.tight_layout() 
        plt.show()
    except Exception as e:
        print(f"Error al generar las gráficas: {e}")

    try:
        df_resultados.to_csv("resultados_bacteria_mejorado.csv", index=False)
        print("Resultados guardados en 'resultados_bacteria_mejorado.csv'")
    except Exception as e:
        print(f"Error al guardar los resultados en CSV: {e}")


    # --- 5. Guardar Mejor Alineamiento ---
    if final_best_idx != -1: # Proceder solo si se obtuvo un índice válido
        print(f"\nGuardando el mejor alineamiento encontrado (Índice {final_best_idx})...")
        best_alignment_sequences = operadorBacterial.get_best_alignment(final_best_idx, poblacion)

        # Verificar que se obtuvieron secuencias y nombres
        if best_alignment_sequences and names:
            if len(best_alignment_sequences) == len(names):
                output_fasta_file = "mejor_alineamiento_bfoa.fasta"
                try:
                    with open(output_fasta_file, "w") as f:
                        for i in range(len(names)):
                            # Limpiar nombre por si acaso
                            clean_name = names[i].strip().replace(" ", "_") 
                            f.write(f">{clean_name}\n")
                            # Escribir secuencia en líneas de 60 caracteres
                            seq = best_alignment_sequences[i]
                            for j in range(0, len(seq), 60):
                                 f.write(seq[j:j+60] + "\n")
                    print(f"Mejor alineamiento guardado en '{output_fasta_file}'")
                except IOError as e:
                    print(f"Error al escribir el archivo FASTA '{output_fasta_file}': {e}")
                except Exception as e:
                     print(f"Error inesperado al guardar el alineamiento: {e}")
            else:
                print(f"Error: El número de secuencias ({len(best_alignment_sequences)}) no coincide con el número de nombres ({len(names)}).")
        elif not best_alignment_sequences:
             print("No se pudo obtener el mejor alineamiento desde la clase bacteria.")
        elif not names:
             print("No se encontraron nombres de secuencia para guardar el archivo FASTA.")
    else:
        print("\nNo se pudo guardar el mejor alineamiento porque no se determinó un índice final válido.")
    # ---------------------------------