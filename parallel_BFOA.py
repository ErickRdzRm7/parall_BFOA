# -*- coding: utf-8 -*-
from __future__ import annotations
import copy
import time
import sys
import random
from multiprocessing import Manager, Value
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from bacteria import bacteria
from typing import Type
from fastaReader import fastaReader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
BASE_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()

FASTA_FILE = Path('/Users/erick/Documents/parall_BFOA/multiFasta.fasta') # Ajusta esto

OUTPUT_DIR = BASE_DIR / 'bfoa_full_results_clean_corrected' # Directorio de salida
OUTPUT_CSV = OUTPUT_DIR / "resultados_bfoa_full_clean_corrected.csv"
OUTPUT_FASTA = OUTPUT_DIR / "mejor_alineamiento_final_clean_corrected.fasta"

# --- BFOA Parameters ---
S = 10                 # Population Size (Nbacterias)
Nc = 10                # Chemotactic steps per reproduction (N_c)
Nre = 10               # Reproduction cycles per E-D (N_re)
Ned = 5                # Elimination-Dispersal cycles (N_ed)
Ns = 4                 # Swim length (N_s): Número máximo de pasos de "nado"

# --- Adaptive Ped Parameters ---
Ped_inicial = 0.25     # Initial Elimination-Dispersal Probability
Ped_final = 0.01       # Final Elimination-Dispersal Probability

# --- Other Parameters ---
Ped_inicial = 0.25     # Initial Elimination-Dispersal Probability
Ped_final = 0.01       # Final Elimination-Dispersal Probability

# --- Other Parameters ---
TUMBO_GAPS = 2 # Gaps to insert/delete during a tumble


D_ATTRACT = 0.3  # Mantener igual que en el mejor conjunto base
W_ATTRACT = 0.1  # Mantener igual que en el mejor conjunto base
H_REPEL = 0.05 # Mantener igual que en el mejor conjunto base
W_REPEL = 0.2  # Mantener igual que en el mejor conjunto base
PRINT_INTERVAL_CH = 5 # Intervalo más frecuente para imprimir progreso quimiotáctico
PLOT_MARKER_SIZE = 3 # Tamaño de los marcadores en las gráficas
PLOT_TITLE_SUFFIX = "BFOA Corregido v1.1" # Sufijo para títulos de gráficas y archivos (incrementamos versión)

# --- Global Variables (Shared via Manager) ---
globalNFE_shared: Optional[Value] = None # type: ignore # Usar Value para el tipo exacto


# --- Function Definitions ---

def load_sequences(fasta_path: Path) -> Tuple[List[List[str]], List[str]]:
    """Loads sequences from a FASTA file."""
    print(f"Cargando secuencias desde '{fasta_path}'...")
    if not fasta_path.is_file():
        print(f"Error fatal: No se encontró el archivo FASTA '{fasta_path}'")
        sys.exit(1)
    try:
        # Ahora fastaReader.__init__ acepta el path
        reader = fastaReader(str(fasta_path))
        sequences_raw = reader.seqs
        names = reader.names
        if not sequences_raw:
            print(f"Error: No se cargaron secuencias desde '{fasta_path}'.")
            # No es fatal si no hay secuencias, pero el BFOA no puede ejecutarse.
            # Decidimos hacerlo fatal ya que sin secuencias no hay alineamiento.
            sys.exit(1)
        print(f"Se cargaron {len(sequences_raw)} secuencias.")
        # Devolver una lista de listas de strings para mutabilidad inicial
        sequences_list = [list(s) for s in sequences_raw]
        return sequences_list, names
    except Exception as e:
        print(f"Error fatal al leer o procesar el archivo FASTA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def setup_multiprocessing(pop_size: int) -> Tuple[Manager, Value, Any, Dict]: # type: ignore
    """Initializes the Multiprocessing Manager and shared variables."""
    print("Inicializando Manager de Multiprocessing...")
    try:
        manager = Manager()
        # Value para contador NFE (seguro con lock implícito o explícito)
        global_nfe = manager.Value('i', 0)
        # Manager.list para la población (almacenará tuplas de tuplas o None)
        poblacion_shared = manager.list([None] * pop_size)
        # Manager.dict para el caché de fitness
        fitness_cache = manager.dict()
        print("Manager inicializado.")
        return manager, global_nfe, poblacion_shared, fitness_cache
    except Exception as e:
        print(f"Error fatal al inicializar multiprocessing.Manager: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def initialize_operator(pop_size: int, cache: Dict, nfe_counter: Value) -> bacteria: # type: ignore
    """Instantiates and configures the bacterial operator."""
    print("Instanciando operador bacterial...")
    try:
        # Pasar las variables compartidas y el tamaño de la población
        op_bacterial = bacteria(pop_size, cache, nfe_counter)
        # Establecer los parámetros de interacción
        op_bacterial.set_interaction_params(0.1, 0.2, 0.1, 0.1)  # Usando los valores previos
        print(f"Parámetros interacción establecidos: dAttr=0.1, wAttr=0.2, hRep=0.1, wRepel=0.1")

        print("Operador bacterial instanciado y configurado.")
        return op_bacterial
    except AttributeError as e:
        print(f"Error fatal: Falta un método o atributo en la clase 'bacteria' ({e}).")
        print("  Verifica que la clase 'bacteria' tenga todos los métodos requeridos.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Error fatal al instanciar la clase 'bacteria': {e}")
        import traceback
        traceback.print_exc()

    """Instantiates and configures the bacterial operator."""
    print("Instanciando operador bacterial...")
    try:
        # Pasar las variables compartidas y el tamaño de la población
        op_bacterial = bacteria(pop_size, cache, nfe_counter)
        # Establecer los parámetros de interacción
        op_bacterial.set_interaction_params(0.1, 0.2, 0.1, 0.1) # Usando los valores previos
        print(f"Parámetros interacción establecidos: dAttr=0.1, wAttr=0.2, hRep=0.1, wRepel=0.1")

        print("Operador bacterial instanciado y configurado.")
        return op_bacterial
    except AttributeError as e:
         print(f"Error fatal: Falta un método o atributo en la clase 'bacteria' ({e}).")
         print("  Verifica que la clase 'bacteria' tenga todos los métodos requeridos.")
         import traceback
         traceback.print_exc()
         sys.exit(1)
    except Exception as e:
        print(f"Error fatal al instanciar la clase 'bacteria': {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def initialize_population(sequences: List[List[str]], population_shared: List, pop_size: int, op_bacterial: bacteria):
    """
    Creates the initial population based on input sequences.
    Each bacterium gets its own mutable copy of the initial sequences,
    then converted to tuple of tuples for the shared list.
    """
    print("Creando población inicial...")
    if len(population_shared) != pop_size:
        print(f"Advertencia: Ajustando tamaño de population_shared a {pop_size}")
        # Redimensionar la lista compartida si es necesario
        while len(population_shared) < pop_size: population_shared.append(None)
        while len(population_shared) > pop_size: population_shared.pop()

    # Crear la población inicial como lista de listas (mutable)
    initial_population_mutable = []
    for i in range(pop_size):
        # Cada bacteria obtiene su propia copia mutable de la estructura de alineamiento inicial
        bacterium_alignment = [list(seq) for seq in sequences]
        initial_population_mutable.append(bacterium_alignment)

    # Asegurar que todos los alineamientos iniciales tengan la misma longitud
    print("Alineando (cuadrando) población inicial...")
    # Cuadra modificará initial_population_mutable in-place
    try:
        op_bacterial.cuadra(len(sequences), initial_population_mutable)
    except Exception as e:
        print(f"Error durante la cuadratura inicial: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    converted_population = []
    for b in initial_population_mutable:
         if b is not None:
              try:
                   converted_population.append(tuple(tuple(seq) for seq in b))
              except Exception as e:
                   print(f"Error convirtiendo alineamiento a tupla de tuplas durante inicialización: {e}. Añadiendo None.")
                   converted_population.append(None)
         else:
              converted_population.append(None)


    # Rellenar o truncar la lista convertida para que coincida con pop_size
    while len(converted_population) < pop_size: converted_population.append(None)
    population_shared[:] = converted_population[:pop_size]

    # Verificar cuántos alineamientos válidos se inicializaron
    valid_initialized = sum(1 for b in population_shared if b is not None)
    if valid_initialized < pop_size:
        print(f"Advertencia: Solo se pudieron inicializar {valid_initialized}/{pop_size} alineamientos válidos.")
        if valid_initialized == 0:
             print("Error fatal: No se pudo inicializar ningún alineamiento válido.")
             sys.exit(1)

    print(f"Población inicial de {pop_size} bacterias creada y cuadrada.")

def evaluate_single_alignment_blosum(op_bacterial: bacteria, alignment_tuple: Optional[Tuple[Tuple[str, ...], ...]]) -> float:
    """Evalúa el score BLOSUM de un solo alineamiento usando el caché y NFE."""
    if alignment_tuple is None:
         return -float('inf') # O 0.0, dependiendo de cómo quieras penalizar None

    try:
        # Usar el método específico para evaluar un solo alineamiento con caché/NFE
        return op_bacterial.evaluaSingleBlosumWithCache(alignment_tuple)
    except Exception as e:
        print(f"Error al evaluar una única alineación (Blosum Score): {e}")
        return -float('inf') # Return a very low fitness to discourage this state


# Ajustada para manejar Manager.list compartido
def evaluate_population(op_bacterial: bacteria, poblacion_shared: List) -> Optional[List[float]]:
    """Evaluates the fitness of the entire population using the bacterial operator."""
    try:
        pop_size_actual = len(poblacion_shared)
        if pop_size_actual == 0 or not any(poblacion_shared):
             # print("Advertencia: Población compartida está vacía o solo contiene None. No se puede evaluar.")
             # Asegurarse de que las tablas internas del operador se reinicien o sean consistentes
             if len(op_bacterial.blosumScore) != op_bacterial.numBacterias: op_bacterial.blosumScore[:] = [0.0] * op_bacterial.numBacterias
             if len(op_bacterial.tablaAtract) != op_bacterial.numBacterias: op_bacterial.tablaAtract[:] = [0.0] * op_bacterial.numBacterias
             if len(op_bacterial.tablaRepel) != op_bacterial.numBacterias: op_bacterial.tablaRepel[:] = [0.0] * op_bacterial.numBacterias
             if len(op_bacterial.tablaInteraction) != op_bacterial.numBacterias: op_bacterial.tablaInteraction[:] = [0.0] * op_bacterial.numBacterias
             # Llenar la tabla de fitness con -inf si no hay bacterias válidas
             op_bacterial.tablaFitness[:] = [-float('inf')] * op_bacterial.numBacterias
             return list(op_bacterial.tablaFitness) # Retornar la lista de -inf
        

        op_bacterial.evaluaBlosumConCache(poblacion_shared)

        # creaTablasAtractRepel espera el Manager.list
        op_bacterial.creaTablasAtractRepel(poblacion_shared)
        # creaTablaInteraction espera el Manager.list
        op_bacterial.creaTablaInteraction(poblacion_shared)
        # creaTablaFitness espera el Manager.list
        op_bacterial.creaTablaFitness(poblacion_shared) # Esto debería generar los valores de fitness total

        fitness_list_result = list(op_bacterial.tablaFitness)
        if len(fitness_list_result) != op_bacterial.numBacterias:
             print(f"Advertencia: Tamaño de la lista de fitness retornada ({len(fitness_list_result)}) no coincide con numBacterias ({op_bacterial.numBacterias}).")
             # Intentar devolver una lista del tamaño esperado con -inf para los faltantes
             while len(fitness_list_result) < op_bacterial.numBacterias: fitness_list_result.append(-float('inf'))
             fitness_list_result = fitness_list_result[:op_bacterial.numBacterias]

        return fitness_list_result

    except AttributeError as e:
        print(f"Error: Falta un método de evaluación en 'bacteria' ({e}). No se puede evaluar.")
        return None
    except Exception as e:
        print(f"Error inesperado durante la evaluación de la población: {e}")
        import traceback
        traceback.print_exc()
        # Asegurarse de que las tablas internas del operador se reinicien o sean consistentes en caso de error
        if len(op_bacterial.blosumScore) != op_bacterial.numBacterias: op_bacterial.blosumScore[:] = [0.0] * op_bacterial.numBacterias
        if len(op_bacterial.tablaAtract) != op_bacterial.numBacterias: op_bacterial.tablaAtract[:] = [0.0] * op_bacterial.numBacterias
        if len(op_bacterial.tablaRepel) != op_bacterial.numBacterias: op_bacterial.tablaRepel[:] = [0.0] * op_bacterial.numBacterias
        if len(op_bacterial.tablaInteraction) != op_bacterial.numBacterias: op_bacterial.tablaInteraction[:] = [0.0] * op_bacterial.numBacterias
        op_bacterial.tablaFitness[:] = [-float('inf')] * op_bacterial.numBacterias # Llenar la tabla de fitness con -inf
        return list(op_bacterial.tablaFitness) # Retornar la lista de -inf para indicar fallo

def run_chemotaxis_step(op_bacterial: bacteria, poblacion_shared: List,
                        num_seqs: int, tumbo_gaps: int, Ns: int, # Added Ns parameter
                        salud_acumulada: List[float], # Lista local, NO compartida
                        step_label: str) -> Tuple[bool, float, int]:
    iter_start_time = time.time()
    global globalNFE_shared

    pop_size = len(poblacion_shared)
    if pop_size == 0 or not any(poblacion_shared):
         print(f"Advertencia: Población vacía o contiene solo None en paso CH {step_label}. Saltando paso.")
         # Asegurarse de que salud_acumulada tenga el tamaño correcto antes de retornar
         while len(salud_acumulada) < pop_size: salud_acumulada.append(-float('inf'))
         while len(salud_acumulada) > pop_size: salud_acumulada.pop()
         # Retornar valores por defecto para paso fallido
         return False, -float('inf'), globalNFE_shared.value if globalNFE_shared else -1
 
    original_total_fitnesses = evaluate_population(op_bacterial, poblacion_shared)
    if not original_total_fitnesses or len(original_total_fitnesses) != pop_size:
        print(f"Error: No se pudieron obtener fitness total de la población actual para comparación en {step_label}.")
        # Asegurarse de que salud_acumulada tenga el tamaño correcto antes de retornar
        while len(salud_acumulada) < pop_size: salud_acumulada.append(-float('inf'))
        while len(salud_acumulada) > pop_size: salud_acumulada.pop()
        return False, -float('inf'), globalNFE_shared.value if globalNFE_shared else -1

    if len(op_bacterial.blosumScore) != pop_size:
         print(f"Error: blosumScore interno ({len(op_bacterial.blosumScore)}) no tiene tamaño {pop_size} antes de CH step {step_label}.")
         # Esto indica un problema serio en evaluate_population o inicialización.
         while len(salud_acumulada) < pop_size: salud_acumulada.append(-float('inf'))
         while len(salud_acumulada) > pop_size: salud_acumulada.pop()
         return False, -float('inf'), globalNFE_shared.value if globalNFE_shared else -1

    original_blosum_scores = list(op_bacterial.blosumScore) # Copia local de los Blosum scores actuales

    # Lista temporal para almacenar los nuevos estados (tupla de tuplas o None) para este paso
    next_poblacion_states: List[Optional[Tuple[Tuple[str, ...], ...]]] = [None] * pop_size

    for i in range(pop_size):
        # Obtener el estado original y sus scores para la bacteria i
        bacteria_original_tuple: Optional[Tuple[Tuple[str, ...], ...]] = poblacion_shared[i]
        original_blosum_score_i = original_blosum_scores[i] # Blosum score original de esta bacteria
        original_total_fitness_i = original_total_fitnesses[i] # Fitness total original de esta bacteria

        if bacteria_original_tuple is None or not np.isfinite(original_total_fitness_i):

            next_poblacion_states[i] = bacteria_original_tuple # Mantener como None o el estado que sea
            # La salud acumulada para esta bacteria no se modificará aquí, se actualizará al final del paso CH.
            continue # Pasar a la siguiente bacteria

        # Convertir a lista de listas para tumbo/cuadra (operaciones mutables)
        try:
            bacteria_original_mutable = [list(seq) for seq in bacteria_original_tuple]
        except Exception as e:
             print(f"Error convirtiendo bacteria {i} a mutable en CH step {step_label}: {e}. Manteniendo estado original.")
             next_poblacion_states[i] = bacteria_original_tuple # Mantener el estado original (tupla de tuplas)
             continue # Pasar a la siguiente bacteria
        
        # El mejor estado mutable y su score BLOSUM encontrado *para esta bacteria* en este paso quimiotáctico.
        best_state_for_bacterium_i_mutable = copy.deepcopy(bacteria_original_mutable)
        best_blosum_score_for_bacterium_i = original_blosum_score_i

        # --- 1. Tumble ---
        # tumbo opera en una copia mutable y devuelve una nueva copia mutable
        tumbled_state_mutable = op_bacterial.tumbo(num_seqs, copy.deepcopy(bacteria_original_mutable), tumbo_gaps)
        # Cuadrar el estado tumbado (modifica in-place la lista pasada)
        op_bacterial.cuadra(num_seqs, [tumbled_state_mutable])

        # Convertir el estado tumbado a tupla de tuplas para evaluación (usando caché/NFE)
        tumbled_state_tuple = tuple(tuple(seq) for seq in tumbled_state_mutable)

        # Evaluar Blosum Score del estado tumbado (usando caché/NFE)
        # Usa el método específico para evaluar un solo alineamiento
        tumbled_blosum_score = evaluate_single_alignment_blosum(op_bacterial, tumbled_state_tuple)

        # Decidir si aceptar el tumbado (basado en Blosum Score temporal)
        # Solo aceptar si el score tumbado es mejor o igual Y finito.
        if np.isfinite(tumbled_blosum_score) and tumbled_blosum_score >= best_blosum_score_for_bacterium_i:
             best_state_for_bacterium_i_mutable = tumbled_state_mutable
             best_blosum_score_for_bacterium_i = tumbled_blosum_score # Actualizar el mejor score Blosum para esta bacteria

             # --- 2. Swim (si se aceptó el tumbado) ---
             current_swim_state_mutable = copy.deepcopy(best_state_for_bacterium_i_mutable)
             current_swim_blosum_score = best_blosum_score_for_bacterium_i # Empezar swim desde el mejor Blosum score hasta ahora

             for s_step in range(Ns):
                # Nadar desde el estado actual del swim
                swum_state_mutable = op_bacterial.tumbo(num_seqs, copy.deepcopy(current_swim_state_mutable), tumbo_gaps)
                op_bacterial.cuadra(num_seqs, [swum_state_mutable])
                swum_state_tuple = tuple(tuple(seq) for seq in swum_state_mutable)

                # Evaluar Blosum Score del estado nadado (usando caché/NFE)
                swum_blosum_score = evaluate_single_alignment_blosum(op_bacterial, swum_state_tuple)

                # Decidir si aceptar el estado nadado (basado en Blosum Score)
                # Solo aceptar si el score nadado es mejor o igual Y finito.
                if np.isfinite(swum_blosum_score) and swum_blosum_score >= current_swim_blosum_score:
                     current_swim_state_mutable = swum_state_mutable
                     current_swim_blosum_score = swum_blosum_score
                     # Actualizar el mejor estado mutable y su score Blosum encontrado para esta bacteria en este paso CH
                     best_state_for_bacterium_i_mutable = current_swim_state_mutable
                     best_blosum_score_for_bacterium_i = current_swim_blosum_score # El nuevo mejor score Blosum para esta bacteria
                else:
                     # Dejar de nadar si no mejora (o es no finito)
                     break
        if best_state_for_bacterium_i_mutable and best_state_for_bacterium_i_mutable[0]:
             next_poblacion_states[i] = tuple(tuple(seq) for seq in best_state_for_bacterium_i_mutable)
        else:
             # Si el mejor estado mutable resultó vacío o inválido, mantener el estado original (puede ser None)
             print(f"Advertencia: Mejor estado mutable para bacteria {i} es inválido al final del CH step. Manteniendo estado original.")
             next_poblacion_states[i] = poblacion_shared[i]

    try:
         # Mantener el estado original si next_poblacion_states[i] es None
         for i in range(pop_size):
              if next_poblacion_states[i] is not None:
                   poblacion_shared[i] = next_poblacion_states[i]
              # Si next_poblacion_states[i] es None, poblacion_shared[i] mantiene su valor previo (que podría ser None).
         # Convertir a lista de listas mutable para cuadrar, excluyendo los None
         poblacion_mutable_for_cuadra = [list(b) for b in poblacion_shared if b is not None]

         if poblacion_mutable_for_cuadra: # Solo cuadrar si hay alineamientos válidos para cuadrar
              op_bacterial.cuadra(num_seqs, poblacion_mutable_for_cuadra)

              # Convertir de vuelta a tupla de tuplas y rellenar la lista compartida
              cuadra_idx = 0
              for i in range(pop_size):
                   if poblacion_shared[i] is not None:
                        # Asignar el alineamiento cuadrado correspondiente desde poblacion_mutable_for_cuadra
                        if cuadra_idx < len(poblacion_mutable_for_cuadra):
                             # Asegurarse de que el alineamiento cuadrado no esté vacío antes de convertir
                             if poblacion_mutable_for_cuadra[cuadra_idx] and poblacion_mutable_for_cuadra[cuadra_idx][0]:
                                  poblacion_shared[i] = tuple(tuple(seq) for seq in poblacion_mutable_for_cuadra[cuadra_idx])
                             else:
                                  print(f"Advertencia: Alineamiento cuadrado para bacteria {i} es inválido. Estableciendo a None.")
                                  poblacion_shared[i] = None # O mantener el estado anterior si posible
                             cuadra_idx += 1
                        else:
                            # Esto no debería pasar si la lógica es correcta y los tamaños coinciden.
                            print(f"Advertencia: Error de índice al re-llenar poblacion_shared con alineamiento cuadrado en índice {i}.")
                            poblacion_shared[i] = None # O mantener el estado anterior si posible
         else:
              # Si no hay alineamientos válidos para cuadrar, la población compartida podría quedar con None
              while len(poblacion_shared) < pop_size: poblacion_shared.append(None)
              poblacion_shared[:] = poblacion_shared[:pop_size] # Truncar si por alguna razón es más larga


    except Exception as e:
         print(f"Error actualizando poblacion_shared o cuadrando al final del paso CH {step_label}: {e}")
         import traceback
         traceback.print_exc()
         while len(salud_acumulada) < pop_size: salud_acumulada.append(-float('inf'))
         while len(salud_acumulada) > pop_size: salud_acumulada.pop()
         return False, historial_mejor_fitness[-1] if historial_mejor_fitness else -float('inf'), globalNFE_shared.value if globalNFE_shared else -1

    current_total_fitnesses = evaluate_population(op_bacterial, poblacion_shared)

    current_nfe_val = globalNFE_shared.value if globalNFE_shared else -1


    if not current_total_fitnesses or len(current_total_fitnesses) != pop_size:
         print(f"Error: No se pudieron obtener fitness total de la población *actualizada* al final del paso CH {step_label}. Salud no actualizada correctamente.")
         # Retornar el último mejor fitness conocido del historial
         best_current_fitness_total = historial_mejor_fitness[-1] if historial_mejor_fitness else -float('inf')
         best_current_idx = -1
         success_status = False # Marcar el paso como fallido
         # Asegurarse de que salud_acumulada tenga el tamaño correcto antes de retornar
         while len(salud_acumulada) < pop_size: salud_acumulada.append(-float('inf'))
         while len(salud_acumulada) > pop_size: salud_acumulada.pop()

    else:
        # Acumular el fitness total de los estados *finales* de este paso CH en la salud de cada bacteria
        if len(salud_acumulada) == len(current_total_fitnesses):
             for i in range(pop_size):
                  if np.isfinite(current_total_fitnesses[i]): # Solo sumar si el fitness es finito
                       salud_acumulada[i] += current_total_fitnesses[i]
                  else:
                       # Si el fitness no es finito, no sumamos. La salud acumulada permanece inalterada.
                       pass
        else:
             print(f"Advertencia: Tamaños inconsistentes de salud_acumulada ({len(salud_acumulada)}) y current_total_fitnesses ({len(current_total_fitnesses)}) al actualizar salud.")
        # Encontrar el mejor fitness TOTAL en la población actualizada para reportar en este paso CH
        # Filtrar valores no finitos antes de encontrar el máximo
        finite_fitnesses = [f for f in current_total_fitnesses if np.isfinite(f)]
        if finite_fitnesses:
             best_current_fitness_total = max(finite_fitnesses)
             # Encontrar el índice del mejor fitness total finito en la lista original (puede ser lento para listas grandes)
             best_current_idx = int(np.argmax(current_total_fitnesses)) # np.argmax funciona bien con -inf correctamente
        else:
             best_current_fitness_total = -float('inf') # Si todos son no finitos
             best_current_idx = -1 # No se encontró un índice válido


        success_status = True # El paso fue exitoso si se obtuvieron fitnesses totales.
    # Imprimir progreso si corresponde
    # Define and initialize total_global_chemotactic_steps if not already defined
    if 'total_global_chemotactic_steps' not in globals():
        total_global_chemotactic_steps = 0  # Initialize to 0 or an appropriate value
    chemotactic_step_index = total_global_chemotactic_steps  # Usar el contador global
    if (chemotactic_step_index % PRINT_INTERVAL_CH == 0) or (chemotactic_step_index == 1): # Imprimir el primer paso también
         iter_time = time.time() - iter_start_time
         print(f"    {step_label} | "
               f"Mejor F (Paso CH Total): {best_current_fitness_total:.4f} (Idx:{best_current_idx}) | "
               f"NFE: {current_nfe_val} | T: {iter_time:.2f}s")

    # Retornar si fue exitoso, el mejor fitness total de este paso, y el NFE actual
    return success_status, best_current_fitness_total, current_nfe_val


def run_reproduction_step(poblacion_shared: List, salud_bacterias: List[float], pop_size: int, op_bacterial: bacteria, step_label: str):
    """Executes the standard BFOA reproduction step."""
    print(f"\n  --- Realizando Reproducción ({step_label}) ---")
    try:
        if len(salud_bacterias) != pop_size:
             print(f"Advertencia: Tamaño de salud_bacterias ({len(salud_bacterias)}) no coincide con pop_size ({pop_size}) en Reproducción. No se puede reproducir correctamente.")
             # No realizar reproducción si los datos de salud son inconsistentes
             return


        bacterias_con_salud = sorted(enumerate(salud_bacterias), key=lambda x: x[1], reverse=True)

        if not bacterias_con_salud:
             print("Error: No hay datos de salud válidos para realizar la reproducción.")
             # Mantener la población compartida como está y salir.
             return


        print(f"    Salud Max: {bacterias_con_salud[0][1]:.2f} (Idx {bacterias_con_salud[0][0]}), "
              f"Salud Min: {bacterias_con_salud[-1][1]:.2f} (Idx {bacterias_con_salud[-1][0]})")

        # Crear la nueva población temporalmente (lista, almacenará tuplas de tuplas)
        nueva_poblacion_temp: List[Optional[Tuple[Tuple[str, ...], ...]]] = [None] * pop_size
        num_padres = pop_size // 2 # La mejor mitad sobrevive y se reproduce
        num_reemplazados = pop_size - num_padres # La peor mitad es reemplazada

        # Copiar la mejor mitad a la nueva población
        for i in range(num_padres):
            original_idx = bacterias_con_salud[i][0]
            # Copiar la bacteria del índice original a la nueva posición
            if 0 <= original_idx < len(poblacion_shared) and poblacion_shared[original_idx] is not None:
                 nueva_poblacion_temp[i] = copy.deepcopy(poblacion_shared[original_idx])
            else:
                 # print(f"Advertencia: Índice original {original_idx} fuera de rango o bacteria es None durante la copia en reproducción. Copiando None.")
                 nueva_poblacion_temp[i] = None # Copiar None si el original es inválido

        # Reemplazar la peor mitad con duplicados de la mejor mitad
        for i in range(num_reemplazados):
            # Usar el operador módulo para ciclar a través de los mejores padres
            parent_rank = i % num_padres # Rango del padre (0 al num_padres-1)
            parent_original_idx = bacterias_con_salud[parent_rank][0] # Índice original del padre

            # Asegurarse de que el padre sea un índice válido y no sea None
            if 0 <= parent_original_idx < len(poblacion_shared) and poblacion_shared[parent_original_idx] is not None:
                nueva_poblacion_temp[num_padres + i] = copy.deepcopy(poblacion_shared[parent_original_idx])
            else:
                 # print(f"Advertencia: Índice padre {parent_original_idx} fuera de rango o padre es None durante la duplicación en reproducción. Copiando None.")
                 nueva_poblacion_temp[num_padres + i] = None # Copiar None si el padre es inválido

        # Actualizar la población compartida con la nueva población generada
        # Asegurarse de que la lista tenga el tamaño correcto (pop_size)
        while len(nueva_poblacion_temp) < pop_size: nueva_poblacion_temp.append(None)
        poblacion_shared[:] = nueva_poblacion_temp[:pop_size]

        valid_reproduced = sum(1 for b in poblacion_shared if b is not None)
        print(f"    Reproducción completada. {num_padres} bacterias mejores duplicadas para reemplazar a {num_reemplazados} peores. {valid_reproduced}/{pop_size} alineamientos válidos en la población.")

    except IndexError:
        print(f"Error de índice durante la reproducción ({step_label}).")
        import traceback
        traceback.print_exc()
    except Exception as e:
         print(f"\nError inesperado durante la reproducción ({step_label}): {e}")
         import traceback
         traceback.print_exc()

# Ajustada para operar sobre Manager.list compartido y usar initialize_random_bacteria
def run_elimination_dispersal_step(poblacion_shared: List, op_bacterial: bacteria, pop_size: int,
                                   elim_prob: float, base_seqs: List[List[str]], num_seqs: int,
                                   step_label: str):
    """Executes the BFOA elimination-dispersal step with elitism."""
    print(f"\n===== Realizando Eliminación-Dispersión ({step_label}, Prob={elim_prob:.4f}) =====")
    try:
        # Evaluar la población actual para aplicar elitismo
        print("    Evaluando fitness actual para elitismo en E-D...")
        current_fitness = evaluate_population(op_bacterial, poblacion_shared)

        best_current_idx = -1
        if current_fitness and len(current_fitness) == pop_size:
            # Encontrar el índice de la mejor bacteria (fitness máximo)
            # Asegurarse de que haya valores finitos antes de argmax
            finite_fitnesses = [f for f in current_fitness if np.isfinite(f)]
            if finite_fitnesses:
                 # np.argmax funciona bien con -inf correctamente
                 best_current_idx = int(np.argmax(current_fitness))
                 print(f"    Mejor bacteria actual (Índice {best_current_idx}, F={current_fitness[best_current_idx]:.4f}) será protegida.")
            else:
                 print("    Advertencia: Todos los fitness son no finitos. Elitismo deshabilitado para este paso.")
                 # best_current_idx permanece -1
        else:
            print("    Advertencia: No se pudo evaluar fitness para elitismo E-D. Elitismo deshabilitado para este paso.")
            # best_current_idx permanece -1

        indices_a_reemplazar = []
        for i in range(pop_size):
            # Omitir a la mejor bacteria si el elitismo está activo y se encontró un mejor índice válido
            if best_current_idx != -1 and i == best_current_idx:
                # print(f"    Protegiendo bacteria {i} (la mejor) de eliminación/dispersión.") # Debug print
                continue
            # Seleccionar bacterias para eliminación/dispersión basándose en la probabilidad
            if random.random() < elim_prob:
                indices_a_reemplazar.append(i)

        bacterias_eliminadas = len(indices_a_reemplazar)

        if bacterias_eliminadas > 0:
            print(f"    {bacterias_eliminadas} bacterias seleccionadas para reemplazo aleatorio.")
            num_reemplazados_exitosos = 0
            for i in indices_a_reemplazar:
                # Asegurarse de que el índice esté dentro del rango
                if 0 <= i < len(poblacion_shared):
                    try:
                        nueva_bacteria_tuple = op_bacterial.initialize_random_bacteria(base_seqs)
                        if nueva_bacteria_tuple is not None:
                             poblacion_shared[i] = nueva_bacteria_tuple
                             num_reemplazados_exitosos += 1
                        else:
                             print(f"    Advertencia: initialize_random_bacteria devolvió None para índice {i}. Se mantiene la bacteria original (o None).")
                             # Si initialize_random_bacteria falla, poblacion_shared[i] mantiene su valor anterior.
                    except AttributeError:
                         print("\n¡¡ERROR FATAL!!: El método 'initialize_random_bacteria' no existe en la clase 'bacteria'.")
                         # Esto debería haber sido detectado antes, pero como seguridad.
                         sys.exit(1)
                    except Exception as e_init:
                         print(f"\n    Error al inicializar bacteria aleatoria para índice {i}: {e_init}. Se mantiene la bacteria original (o None).")
                         import traceback
                         traceback.print_exc()
                         # Si initialize_random_bacteria falla, poblacion_shared[i] mantiene su valor anterior.
                else:
                     print(f"Advertencia: Índice {i} fuera de rango durante Eliminación-Dispersión. Saltando reemplazo.")


            print(f"    {num_reemplazados_exitosos}/{bacterias_eliminadas} bacterias seleccionadas reemplazadas exitosamente.")
            print("    Re-evaluando población post-eliminación/dispersión...")
            # La cuadratura de la población completa se realizará al inicio del próximo ciclo de quimiotaxis.
            # Solo evaluamos el fitness aquí para reportar el progreso.
            fitness_post_ed = evaluate_population(op_bacterial, poblacion_shared)
            if fitness_post_ed:
                  # Filtrar valores no finitos antes de encontrar el máximo para el reporte
                 finite_fitnesses = [f for f in fitness_post_ed if np.isfinite(f)]
                 if finite_fitnesses:
                      best_fitness_post_ed = max(finite_fitnesses)
                      print(f"    Mejor fitness post-E/D: {best_fitness_post_ed:.4f}")
                 else:
                      print("    Advertencia: Todos los fitness son no finitos post-E/D.")
            else:
                 print("Advertencia: No se pudo obtener fitness de la población post-E/D.")

        else:
            print("    No se eliminaron bacterias en este ciclo (o solo se protegió a la mejor).")

    except Exception as e:
        print(f"\nError durante la eliminación-dispersal ({step_label}): {e}")
        import traceback
        traceback.print_exc()


def process_final_results(history_fitness: List[float], history_nfe: List[float],
                         poblacion_final: List, op_bacterial: bacteria, pop_size: int
                         ) -> Tuple[float, int, float, pd.DataFrame]:
    print("\n--- Procesando Resultados Finales ---")

    final_nfe = history_nfe[-1] if history_nfe else 0
    # El mejor fitness global es el máximo en todo el historial registrado de mejores fitness por paso CH
    best_overall_fitness_history = max(history_fitness) if history_fitness else -float('inf')
    total_steps_recorded = len(history_fitness)
    print("    Evaluando fitness de la población final para obtener el mejor alineamiento...")
    final_fitnesses_pop = evaluate_population(op_bacterial, poblacion_final)

    final_best_fitness_pop = -float('inf')
    final_best_idx_pop = -1
    if final_fitnesses_pop is not None and len(final_fitnesses_pop) == pop_size:
        # Encontrar el mejor fitness finito y su índice
        finite_fitnesses_pop = [f for f in final_fitnesses_pop if np.isfinite(f)]
        if finite_fitnesses_pop:
             final_best_fitness_pop = max(finite_fitnesses_pop)
             # Encontrar el índice del mejor fitness total finito en la lista original
             # np.argmax funciona bien con -inf correctamente, así que podemos usarlo en la lista original
             final_best_idx_pop = int(np.argmax(final_fitnesses_pop))
             print(f"Mejor Fitness (Población Final): {final_best_fitness_pop:.4f} (Índice {final_best_idx_pop})")
        else:
             print("Advertencia: Todos los fitness de la población final son no finitos.")
             final_best_fitness_pop = -float('inf') # O el último mejor del historial?
             final_best_idx_pop = -1

    else:
        print("Advertencia: No se pudo obtener fitness de la población final. Usando el último mejor registrado del historial.")
        # Si no se pudo evaluar la población final, usamos el mejor del historial y no podemos determinar el índice final.
        final_best_fitness_pop = history_fitness[-1] if history_fitness else -float('inf')
        final_best_idx_pop = -1 # No podemos asegurar el índice correcto en la población final sin re-evaluación exitosa.

    print(f"Mejor Fitness (Historial Global): {best_overall_fitness_history:.4f}")
    print(f"Total NFE: {final_nfe}")

    # Crear DataFrame para guardar los resultados paso a paso
    min_len = min(len(history_fitness), len(history_nfe))
    df_results = pd.DataFrame({
        'Paso_Quimiotactico_Global': list(range(min_len)),
        'Mejor_Fitness_Paso': history_fitness[:min_len],
        'NFE_Acumulado': history_nfe[:min_len]
    })

    print(f"\nResumen de Resultados (Total Pasos Quimiotácticos Registrados: {min_len -1 if min_len > 0 else 0}):")
    print(df_results.head())
    print("...")
    print(df_results.tail())
    return final_best_fitness_pop, final_best_idx_pop, best_overall_fitness_history, df_results


def plot_results(df: pd.DataFrame, final_nfe_val: int, best_overall_fit: float, title_suffix: str):
    """Generates plots showing the progress of fitness and NFE."""
    print("Generando gráficas...")
    if df.empty:
        print("No hay datos para graficar.")
        return

    try:
        plt.figure(figsize=(14, 7))

        # Gráfica de Mejor Fitness a lo largo del tiempo
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(df['Paso_Quimiotactico_Global'], df['Mejor_Fitness_Paso'], marker='o', linestyle='-', color='b', markersize=PLOT_MARKER_SIZE, label='Mejor Fitness en el Paso CH')
        # Añadir línea horizontal para el mejor fitness global encontrado
        if np.isfinite(best_overall_fit):
             ax1.axhline(y=best_overall_fit, color='g', linestyle='--', label=f'Mejor Global ({best_overall_fit:.4f})')
        ax1.set_title(f"Progreso Mejor Fitness ({title_suffix})")
        ax1.set_xlabel(f"Paso Quimiotáctico Global") # El eje X es el índice del paso CH
        ax1.set_ylabel("Mejor Fitness (J) encontrado en el Paso CH")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Gráfica de NFE Acumulado
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(df['Paso_Quimiotactico_Global'], df['NFE_Acumulado'], marker='s', linestyle='--', color='r', markersize=PLOT_MARKER_SIZE)
        ax2.set_title(f"NFE Acumulado ({title_suffix})\nNFE Total Final: {final_nfe_val}")
        ax2.set_xlabel(f"Paso Quimiotáctico Global") # El eje X es el índice del paso CH
        ax2.set_ylabel("NFE Acumulado")
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout() # Ajustar el layout para evitar solapamiento
        # Crear el directorio de salida si no existe
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Generar nombre de archivo seguro
        plot_filename = OUTPUT_DIR / f"plot_fitness_nfe_{title_suffix.replace(' ', '_').replace('.', '').replace('/', '_')}.png" # Reemplazar / también por seguridad
        plt.savefig(plot_filename)
        print(f"Gráfica guardada en: {plot_filename}")
        # plt.show() # Mostrar la gráfica (puede ser bloqueante)
        # print("Gráficas mostradas.")

    except Exception as e:
        print(f"Error al generar o guardar las gráficas: {e}")
        import traceback
        traceback.print_exc()


def save_results_to_csv(df: pd.DataFrame, csv_path: Path):
    """Saves the results DataFrame to a CSV file."""
    print(f"Guardando resultados detallados en '{csv_path}'...")
    try:
        # Crear el directorio padre si no existe
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print("Resultados guardados en CSV.")
    except Exception as e:
        print(f"Error al guardar resultados en CSV: {e}")
        import traceback
        traceback.print_exc()

def save_best_alignment_to_fasta(op_bacterial: bacteria, best_idx: int, poblacion: List,
                                 seq_names: List[str], fasta_path: Path):
    """Saves the best alignment found in the final population to a FASTA file."""
    # poblacion es el Manager.list compartido (tupla de tuplas)

    if best_idx == -1 or best_idx >= len(poblacion):
        print(f"\nAdvertencia: Índice del mejor alineamiento ({best_idx}) inválido o fuera de rango ({len(poblacion)}). No se guardará el FASTA.")
        return

    # Obtener el alineamiento (puede ser None)
    alignment_structure: Optional[Tuple[Tuple[str, ...], ...]] = poblacion[best_idx]

    if alignment_structure is None or not alignment_structure:
        print(f"Advertencia: El alineamiento para el índice {best_idx} es None o está vacío. No se guardará el FASTA.")
        return
    if not seq_names:
        print("Advertencia: No se encontraron nombres de secuencia para guardar en FASTA. No se guardará el FASTA.")
        return
    if len(alignment_structure) != len(seq_names):
        print(f"Advertencia: Discrepancia en número de secuencias ({len(alignment_structure)}) vs nombres ({len(seq_names)}). Intentando guardar las secuencias disponibles.")
        # Intentar guardar las secuencias disponibles con nombres si es posible
        num_to_save = min(len(alignment_structure), len(seq_names))
        if num_to_save > 0:
            print(f"Guardando las primeras {num_to_save} secuencias.")
            alignment_structure_subset = alignment_structure[:num_to_save]
            seq_names_subset = seq_names[:num_to_save]
        else:
             print("No hay secuencias o nombres suficientes para guardar después del ajuste.")
             return # No guardar si después del ajuste no hay datos válidos

    else: # Los tamaños coinciden, guardar todo
        alignment_structure_subset = alignment_structure
        seq_names_subset = seq_names


    print(f"\nGuardando mejor alineamiento de población final (Índice {best_idx}) en '{fasta_path}'...")
    try:
        # Crear el directorio padre si no existe
        fasta_path.parent.mkdir(parents=True, exist_ok=True)

        with open(fasta_path, "w") as f:
            for i, name in enumerate(seq_names_subset):
                # Limpiar nombre para el encabezado FASTA
                clean_name = str(name).strip().replace(" ", "_").replace(">", "")
                if not clean_name: # Asegurarse de que el nombre no esté vacío después de limpiar
                     clean_name = f"sequence_{i+1}" # Nombre por defecto si está vacío

                f.write(f">{clean_name}\n")

                # Convertir la tupla de caracteres a string
                seq_tuple_or_list = alignment_structure_subset[i] # Puede ser tupla o lista
                # Asegurarse de que es iterable de caracteres
                if not isinstance(seq_tuple_or_list, (list, tuple)):
                     print(f"Advertencia: Elemento inesperado en alineamiento[{i}]. Saltando guardar secuencia.")
                     continue

                seq_str = "".join(seq_tuple_or_list)

                # Escribir la secuencia en líneas de 60 caracteres
                for j in range(0, len(seq_str), 60):
                    f.write(seq_str[j:j+60] + "\n")

        print("Mejor alineamiento guardado en formato FASTA.")

    except IndexError:
         print(f"Error: Error de índice al intentar acceder a alineamiento o nombres al guardar FASTA para índice {best_idx}.")
         import traceback
         traceback.print_exc()
    except AttributeError:
         print("Error: Problema al acceder o procesar la estructura del alineamiento al intentar guardar FASTA.")
         import traceback
         traceback.print_exc()
    except Exception as e:
         print(f"Error inesperado al guardar el mejor alineamiento: {e}")
         import traceback
         traceback.print_exc()

# --- Entry Point ---
if __name__ == "__main__":
    # Inicializar historial_mejor_fitness como una lista vacía
    historial_mejor_fitness = []

    # Iniciar la ejecución principal del algoritmo
    def main():
        """Main function to execute the BFOA algorithm."""
        print("Starting BFOA algorithm...")

        # Load sequences from the FASTA file
        sequences, seq_names = load_sequences(FASTA_FILE)

        # Setup multiprocessing shared variables
        manager, global_nfe, poblacion_shared, fitness_cache = setup_multiprocessing(S)
        global globalNFE_shared
        globalNFE_shared = global_nfe

        # Initialize the bacterial operator
        op_bacterial = initialize_operator(S, fitness_cache, global_nfe)

        # Initialize the population
        initialize_population(sequences, poblacion_shared, S, op_bacterial)

        # Initialize variables for the algorithm
        salud_acumulada = [0.0] * S
        historial_mejor_fitness = []
        historial_nfe = []

        # Main BFOA loop
        for ed_cycle in range(Ned):
            print(f"\n=== Elimination-Dispersal Cycle {ed_cycle + 1}/{Ned} ===")
            for re_cycle in range(Nre):
                print(f"\n--- Reproduction Cycle {re_cycle + 1}/{Nre} ---")
                for ch_step in range(Nc):
                    step_label = f"ED{ed_cycle + 1}_RE{re_cycle + 1}_CH{ch_step + 1}"
                    success, best_fitness, current_nfe = run_chemotaxis_step(
                        op_bacterial, poblacion_shared, len(sequences), TUMBO_GAPS, Ns, salud_acumulada, step_label
                    )
                    if success:
                        historial_mejor_fitness.append(best_fitness)
                        historial_nfe.append(current_nfe)
                    else:
                        print(f"Step {step_label} failed. Skipping to next step.")
                        continue
        # Perform reproduction only if there is a significant improvement in fitness
        if len(historial_mejor_fitness) > 1 and best_fitness > historial_mejor_fitness[-2]:
            run_reproduction_step(poblacion_shared, salud_acumulada, S, op_bacterial, f"ED{ed_cycle + 1}_RE{re_cycle + 1}")
        else:
            print(f"Skipping reproduction for {step_label} due to insufficient fitness improvement.")

        # Perform elimination-dispersal with adaptive probability
        Ped_actual = max(Ped_final, Ped_inicial - (ed_cycle / (Ned - 1)) * (Ped_inicial - Ped_final))
        run_elimination_dispersal_step(poblacion_shared, op_bacterial, S, Ped_actual, sequences, len(sequences), f"ED{ed_cycle + 1}")

        # Process final results only once at the end of all cycles
        if ed_cycle == Ned - 1 and re_cycle == Nre - 1:
            final_best_fitness, final_best_idx, best_overall_fitness, df_results = process_final_results(
                historial_mejor_fitness, historial_nfe, poblacion_shared, op_bacterial, S
            )

            # Save results
            save_results_to_csv(df_results, OUTPUT_CSV)
            save_best_alignment_to_fasta(op_bacterial, final_best_idx, poblacion_shared, seq_names, OUTPUT_FASTA)

            # Plot results
            plot_results(df_results, historial_nfe[-1] if historial_nfe else 0, best_overall_fitness, PLOT_TITLE_SUFFIX)
            print("BFOA algorithm completed.")

    if __name__ == "__main__":
        main()