import copy
import math
from multiprocessing import Manager, Pool, Lock # Añadir Lock si es necesario para el caché
from evaluadorBlosum import evaluadorBlosum
import numpy as np # Usar np en lugar de numpy
import random
import concurrent.futures # Asegúrate que esta importación esté presente

# --- Función auxiliar para convertir alineamiento a hashable ---
# (Puede estar fuera o dentro de la clase)
def alignment_to_hashable(alignment):
    # (Código sin cambios)
    if isinstance(alignment, tuple): 
        return alignment
    return tuple(tuple(seq) for seq in alignment)

class bacteria():
    # (__init__ sin cambios)
    def __init__(self, numBacterias, fitness_cache, nfe_counter):
        manager = Manager() 
        self.numBacterias = numBacterias
        self.blosumScore = manager.list([0.0] * numBacterias) 
        self.tablaAtract = manager.list([0.0] * numBacterias)
        self.tablaRepel = manager.list([0.0] * numBacterias)
        self.tablaInteraction = manager.list([0.0] * numBacterias)
        self.tablaFitness = manager.list([0.0] * numBacterias)
        self.fitness_cache = fitness_cache 
        self.globalNFE = nfe_counter 
        self.evaluador = evaluadorBlosum() 
        # self.cache_lock = Lock()

    # (cuadra sin cambios)
    def cuadra(self, numSec, poblacion):
        # ... (código anterior de cuadra) ...
        try:
            poblacion_list = [list(b) for b in poblacion]
        except TypeError: 
             poblacion_list = poblacion 
        maxLen = 0
        for i in range(len(poblacion_list)):
            bacterTmp = poblacion_list[i]
            if not bacterTmp: continue 
            try:
                current_max = max(len(s) for s in bacterTmp if s) if any(bacterTmp) else 0
            except ValueError: 
                 current_max = 0
            except Exception as e:
                 print(f"Error calculando maxLen en cuadra para bacteria {i}: {e}")
                 current_max = 0 
            if current_max > maxLen:
                maxLen = current_max
        for i in range(len(poblacion_list)):
            bacterTmp = poblacion_list[i]
            if not bacterTmp: continue
            num_seqs_in_bact = len(bacterTmp) 
            for t in range(num_seqs_in_bact): 
                 if not isinstance(bacterTmp[t], list):
                     bacterTmp[t] = list(bacterTmp[t])
                 gap_count = maxLen - len(bacterTmp[t])
                 if gap_count > 0:
                     bacterTmp[t].extend(["-"] * gap_count)
            poblacion[i] = tuple(tuple(s) for s in bacterTmp) 

    # (tumbo sin cambios)
    def tumbo(self, numSec, poblacion, numGaps):
        # ... (código anterior de tumbo) ...
        for i in range(len(poblacion)):
            try:
                 bacterTmp = [list(seq) for seq in poblacion[i]] 
            except TypeError:
                 print(f"Error convirtiendo bacteria {i} a lista de listas en tumbo. Saltando.")
                 continue 
            for _ in range(numGaps):
                if not bacterTmp: continue 
                try:
                    seqnum = random.randint(0, len(bacterTmp)-1)
                    pos = random.randint(0, len(bacterTmp[seqnum]))
                    bacterTmp[seqnum].insert(pos, "-")
                except IndexError:
                     print(f"Error de índice en tumbo para bacteria {i}. Saltando inserción de gap.")
                     continue 
                except Exception as e:
                     print(f"Error inesperado en tumbo para bacteria {i}: {e}. Saltando inserción de gap.")
                     continue
            poblacion[i] = tuple(tuple(s) for s in bacterTmp)

    # (_get_pairs_for_eval sin cambios)
    def _get_pairs_for_eval(self, alignment):
        # ... (código anterior de _get_pairs_for_eval) ...
        pares = []
        if not alignment or not alignment[0]: 
            return [] 
        try:
            num_seqs = len(alignment)
            len_seq = len(alignment[0]) 
            if not all(len(s) == len_seq for s in alignment):
                 print(f"Advertencia: Longitudes inconsistentes detectadas en _get_pairs_for_eval. Longitudes: {[len(s) for s in alignment]}")
            for col_idx in range(len_seq):
                columna = []
                for row_idx in range(num_seqs):
                     try:
                          columna.append(alignment[row_idx][col_idx])
                     except IndexError:
                          print(f"Error de índice recuperando columna {col_idx}, fila {row_idx}. Usando '-' por defecto.")
                          columna.append("-") 
                for i in range(num_seqs):
                    for j in range(i + 1, num_seqs):
                        par = tuple(sorted((columna[i], columna[j])))
                        pares.append(par)
        except Exception as e:
             print(f"Error inesperado en _get_pairs_for_eval: {e}")
             return [] 
        return pares 

    # (_calculate_blosum_score sin cambios)
    def _calculate_blosum_score(self, alignment_hashable):
        # ... (código anterior de _calculate_blosum_score) ...
         alignment_list = [list(s) for s in alignment_hashable]
         pares = self._get_pairs_for_eval(alignment_list)
         score = 0
         for par in pares:
             try:
                  score += self.evaluador.getScore(par[0], par[1])
             except Exception as e:
                  print(f"Error obteniendo score para par {par}: {e}")
                  score += -10 
         return score

    # (evaluaBlosumConCache sin cambios)
    def evaluaBlosumConCache(self, poblacion):
        # ... (código anterior de evaluaBlosumConCache) ...
        scores = [0.0] * len(poblacion)
        for i in range(len(poblacion)):
            try:
                alignment = poblacion[i] 
                if not alignment or not all(alignment):
                    print(f"Advertencia: Alineamiento inválido o vacío para bacteria {i}. Usando score 0.")
                    scores[i] = 0.0 
                    continue 
            except Exception as e:
                 print(f"Error accediendo a la bacteria {i} en evaluaBlosumConCache: {e}. Usando score 0.")
                 scores[i] = 0.0
                 continue
            try:
                if alignment in self.fitness_cache:
                    scores[i] = self.fitness_cache[alignment]
                else:
                    score = self._calculate_blosum_score(alignment)
                    self.globalNFE.value += 1 
                    self.fitness_cache[alignment] = score 
                    scores[i] = score
            except TypeError as e:
                 print(f"Error de tipo (posiblemente no hashable) con alineamiento para bacteria {i}: {e}")
                 scores[i] = self._calculate_blosum_score(alignment) 
            except Exception as e:
                 print(f"Error inesperado durante cache/cálculo para bacteria {i}: {e}")
                 scores[i] = 0.0 
        try:
            self.blosumScore[:] = scores
        except Exception as e:
             print(f"Error actualizando self.blosumScore: {e}")

    # (evaluaBlosumInicial sin cambios)
    def evaluaBlosumInicial(self, poblacion):
        print("  (Llamando a evaluaBlosumConCache para evaluación inicial)")
        self.evaluaBlosumConCache(poblacion)

    # --- Métodos de Interacción ---
    
    # (compute_diff con la fórmula CORREGIDA exp(-w*diff))
    def compute_diff(self, args):
        # ... (código CORREGIDO de compute_diff de la respuesta anterior) ...
        indexBacteria, otherBlosumScore, allBlosumScores, d, w, otherFitness, currentFitness = args
        interaction_value = 0.0 
        try:
            score_self = allBlosumScores[indexBacteria]
            score_other = otherBlosumScore
            if score_self is None or score_other is None:
                 diff = float('inf') 
            else:
                 diff = (score_self - score_other) ** 2.0
            if w < 0:
                 print(f"Advertencia: 'w' negativo ({w}) pasado a compute_diff. Usando valor absoluto.")
                 w = abs(w)
            elif w == 0:
                 interaction_value = d 
                 return interaction_value 
            if diff == float('inf'):
                interaction_value = 0.0
            elif diff == 0:
                interaction_value = d
            else:
                exponent = -w * diff
                interaction_value = d * np.exp(exponent)
            if not np.isfinite(interaction_value):
                 interaction_value = 0.0
        except IndexError:
             print(f"Error de índice en compute_diff accediendo a scores. Usando interacción 0.")
             interaction_value = 0.0
        except OverflowError:
             print(f"Error de overflow en compute_diff (d={d}, w={w}, diff={diff}). Usando 0.")
             interaction_value = 0.0
        except Exception as e:
             print(f"Error inesperado en compute_diff: {e}")
             interaction_value = 0.0 
        return interaction_value

    # (compute_cell_interaction sin cambios)
    def compute_cell_interaction(self, indexBacteria, d, w, atracTrue, poblacion):
        # ... (código anterior de compute_cell_interaction) ...
        total = 0.0 
        try:
            if indexBacteria >= len(self.tablaFitness) or indexBacteria >= len(self.blosumScore):
                 print(f"Error: Índice {indexBacteria} fuera de rango para scores en compute_cell_interaction.")
                 return 
            currentFitness = self.tablaFitness[indexBacteria] 
            currentBlosum = self.blosumScore[indexBacteria]
            num_bacterias = len(self.blosumScore) 
            if num_bacterias == 0: return 
            blosum_scores_list = list(self.blosumScore)
            fitness_list = list(self.tablaFitness)
            args_list = [
                (indexBacteria, blosum_scores_list[i], blosum_scores_list, d, w, fitness_list[i], currentFitness) 
                for i in range(num_bacterias) 
            ]
            if num_bacterias < 4: 
                 results = [self.compute_diff(arg) for arg in args_list]
            else:
                try:
                    with Pool(processes=min(4, num_bacterias)) as pool: 
                        results = pool.map(self.compute_diff, args_list)
                except Exception as e:
                    print(f"Error en Pool para interacción (Bacteria {indexBacteria}): {e}")
                    print("  (Intentando cálculo serial como fallback...)")
                    results = [self.compute_diff(arg) for arg in args_list] 
            total = sum(results)
            if not np.isfinite(total):
                 print(f"Advertencia: Suma de interacción no finita para bacteria {indexBacteria}. Usando 0.")
                 total = 0.0 
        except Exception as e:
            print(f"Error inesperado en compute_cell_interaction para bacteria {indexBacteria}: {e}")
            total = 0.0 
        try:
            if atracTrue: 
                self.tablaAtract[indexBacteria] = total
            else:
                self.tablaRepel[indexBacteria] = total
        except IndexError:
             print(f"Error de índice al actualizar tabla Atract/Repel para bacteria {indexBacteria}.")
        except Exception as e:
             print(f"Error inesperado actualizando tabla Atract/Repel para bacteria {indexBacteria}: {e}")


    # (creaTablaAtract sin cambios)
    def creaTablaAtract(self, poblacion, d, w):
        # ... (código anterior de creaTablaAtract) ...
        pop_list = list(poblacion) 
        for indexBacteria in range(len(pop_list)):
             self.compute_cell_interaction(indexBacteria, d, w, True, pop_list) 

    # (creaTablaRepel sin cambios)
    def creaTablaRepel(self, poblacion, d, w):
        # ... (código anterior de creaTablaRepel) ...
         pop_list = list(poblacion) 
         for indexBacteria in range(len(pop_list)):
              self.compute_cell_interaction(indexBacteria, d, w, False, pop_list)

    # (creaTablasAtractRepel sin cambios - usando versión serial)
    def creaTablasAtractRepel(self, poblacion, dAttr, wAttr, hRep, wRepel):
        # ... (código anterior de creaTablasAtractRepel - versión serial) ...
         print("    Calculando Atracción/Repulsión (Serial)...")
         try:
             self.creaTablaAtract(poblacion, dAttr, wAttr) 
             self.creaTablaRepel(poblacion, hRep, wRepel) 
             print("    Cálculos de Atracción/Repulsión completados.")
         except Exception as e:
             print(f"Error durante el cálculo serial de Atracción/Repulsión: {e}")
             print("Warning: Fitness podría ser incorrecto. Tablas de interacción pueden estar incompletas.")


    # (creaTablaInteraction sin cambios)
    def creaTablaInteraction(self):
        # ... (código anterior de creaTablaInteraction) ...
        try:
            len_atract = len(self.tablaAtract)
            len_repel = len(self.tablaRepel)
            target_len = self.numBacterias
            interaction_results = [0.0] * target_len
            for i in range(target_len):
                atract = self.tablaAtract[i] if i < len_atract else 0.0
                repel = self.tablaRepel[i] if i < len_repel else 0.0
                sum_interact = atract + repel
                if not np.isfinite(sum_interact):
                     sum_interact = 0.0
                interaction_results[i] = sum_interact
            self.tablaInteraction[:] = interaction_results
        except Exception as e:
            print(f"Error en creaTablaInteraction: {e}")
            self.tablaInteraction[:] = [0.0] * self.numBacterias

    # (creaTablaFitness sin cambios)
    def creaTablaFitness(self):
        # ... (código anterior de creaTablaFitness) ...
         try:
             len_blosum = len(self.blosumScore)
             len_interact = len(self.tablaInteraction)
             target_len = self.numBacterias
             fitness_results = [0.0] * target_len
             for i in range(target_len):
                 valorBlsm = self.blosumScore[i] if i < len_blosum else 0.0
                 valorInteract = self.tablaInteraction[i] if i < len_interact else 0.0
                 if not np.isfinite(valorBlsm):
                      valorBlsm = 0.0
                 if not np.isfinite(valorInteract):
                      valorInteract = 0.0
                 fitness_results[i] = valorBlsm + valorInteract
             self.tablaFitness[:] = fitness_results
         except Exception as e:
              print(f"Error en creaTablaFitness: {e}")
              self.tablaFitness[:] = [-float('inf')] * self.numBacterias

    # (getNFE sin cambios)
    def getNFE(self):
        # ... (código anterior de getNFE) ...
        try:
            return self.globalNFE.value
        except Exception as e:
             print(f"Error accediendo a NFE global: {e}")
             return -1 

    # (obtieneBest sin cambios)
    def obtieneBest(self, current_global_nfe):
        # ... (código anterior de obtieneBest) ...
         bestIdx = -1
         best_fitness_val = -float('inf') 
         try:
             fitness_list = list(self.tablaFitness) 
             if not fitness_list: 
                  print("Advertencia: tablaFitness vacía en obtieneBest.")
                  return -1, -float('inf')
             bestIdx = 0
             for idx_init, f_init in enumerate(fitness_list):
                  if np.isfinite(f_init):
                       bestIdx = idx_init
                       best_fitness_val = f_init
                       break
             else: 
                  print("Advertencia: No se encontró ningún fitness finito válido.")
                  return 0, fitness_list[0] if fitness_list else -float('inf') 
             for i in range(bestIdx + 1, len(fitness_list)):
                  current_val = fitness_list[i]
                  if np.isfinite(current_val) and current_val > best_fitness_val:
                      best_fitness_val = current_val
                      bestIdx = i
             best_blosum = self.blosumScore[bestIdx] if bestIdx < len(self.blosumScore) else 'N/A'
             best_interact = self.tablaInteraction[bestIdx] if bestIdx < len(self.tablaInteraction) else 'N/A'
             if bestIdx != -1:
                  print(f"--- Mejor Bacteria Actual: Índice={bestIdx}, Fitness={best_fitness_val:.4f} (Blosum={best_blosum}, Interact={best_interact}), NFE Total={current_global_nfe} ---")
         except Exception as e:
              print(f"Error en obtieneBest: {e}")
              return -1, -float('inf') 
         return bestIdx, best_fitness_val

    # (replaceWorst sin cambios)
    def replaceWorst(self, poblacion, bestIdx):
        # ... (código anterior de replaceWorst) ...
        worstIdx = -1
        worst_fitness_val = float('inf') 
        try:
            fitness_list = list(self.tablaFitness)
            if not fitness_list or bestIdx < 0 or bestIdx >= len(poblacion): 
                 print("Advertencia: No se puede ejecutar replaceWorst (datos inválidos o bestIdx fuera de rango).")
                 return 
            worstIdx = 0
            for idx_init, f_init in enumerate(fitness_list):
                 if np.isfinite(f_init):
                      worstIdx = idx_init
                      worst_fitness_val = f_init
                      break
            else: 
                 print("Advertencia: No se encontró ningún fitness finito válido para determinar el peor.")
                 return 
            for i in range(worstIdx + 1, len(fitness_list)):
                 current_val = fitness_list[i]
                 if np.isfinite(current_val) and current_val < worst_fitness_val:
                     worst_fitness_val = current_val
                     worstIdx = i
            if worstIdx != -1 and worstIdx != bestIdx: 
                print(f"    Reemplazando peor bacteria (Índice {worstIdx}, Fitness {worst_fitness_val:.4f}) con copia de la mejor (Índice {bestIdx}).")
                poblacion[worstIdx] = poblacion[bestIdx] 
                try:
                    self.blosumScore[worstIdx] = self.blosumScore[bestIdx]
                    self.tablaFitness[worstIdx] = self.tablaFitness[bestIdx]
                    self.tablaInteraction[worstIdx] = self.tablaInteraction[bestIdx]
                    self.tablaAtract[worstIdx] = self.tablaAtract[bestIdx]
                    self.tablaRepel[worstIdx] = self.tablaRepel[bestIdx]
                except IndexError:
                     print(f"Advertencia: Error de índice al copiar scores durante replaceWorst para índice {worstIdx}.")
                except Exception as e:
                     print(f"Error inesperado copiando scores en replaceWorst: {e}")
            elif worstIdx == bestIdx:
                print(f"    La mejor bacteria (Índice {bestIdx}) es también la peor. No se realiza reemplazo.")
        except Exception as e:
             print(f"Error en replaceWorst: {e}")


    # --- 5. MÉTODO AÑADIDO PARA OBTENER EL MEJOR ALINEAMIENTO ---
    def get_best_alignment(self, bestIdx, poblacion):
        """Recupera el alineamiento de la mejor bacteria y lo formatea como lista de strings."""
        # Verificar si el índice es válido
        if 0 <= bestIdx < len(poblacion):
            try:
                # El alineamiento se almacena como tupla de tuplas de caracteres
                alignment_tuple = poblacion[bestIdx] 
                # Convertir a lista de strings para salida/visualización
                alignment_strings = ["".join(seq) for seq in alignment_tuple]
                return alignment_strings
            except Exception as e:
                print(f"Error recuperando/formateando el mejor alineamiento para índice {bestIdx}: {e}")
                return None # Devolver None si hay error
        else:
            print(f"Error: Índice del mejor ({bestIdx}) fuera de rango para la población.")
            return None # Devolver None si el índice no es válido
    # -----------------------------------------------------------