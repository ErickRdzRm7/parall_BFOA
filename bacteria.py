import copy
import math
import random
import numpy as np
# Eliminar Pool y Lock ya que no se usarán en compute_cell_interaction
from multiprocessing import Manager, Value # Mantener Manager y Value
from typing import List, Tuple, Dict, Any, Optional

# Asegúrate de que evaluadorBlosum esté en el mismo directorio o accesible en el PATH
from evaluadorBlosum import evaluadorBlosum

def alignment_to_hashable(alignment):
    if isinstance(alignment, tuple) and all(isinstance(seq, tuple) for seq in alignment):
        return alignment # Ya está en formato hashable (tupla de tuplas)

    if alignment is None:
        return None
    try:
        # Asegurarse de que cada secuencia sea iterable
        return tuple(tuple(seq) for seq in alignment if isinstance(seq, (list, tuple)))
    except Exception as e:
        print(f"Error al convertir alineamiento a hashable: {e}")
        # Devolver None o una tupla vacía en caso de error
        return tuple()


class bacteria():

    def __init__(self, numBacterias: int, fitness_cache: Dict, nfe_counter: Any):
        # Usar Manager para listas/diccionarios compartidos
        manager = Manager()
        self.numBacterias = numBacterias
        # Listas compartidas para almacenar resultados por bacteria
        self.blosumScore: List[float] = manager.list([0.0] * numBacterias)
        self.tablaAtract: List[float] = manager.list([0.0] * numBacterias)
        self.tablaRepel: List[float] = manager.list([0.0] * numBacterias)
        self.tablaInteraction: List[float] = manager.list([0.0] * numBacterias)
        self.tablaFitness: List[float] = manager.list([0.0] * numBacterias)

        # Caché de fitness compartido y contador NFE global
        self.fitness_cache = fitness_cache # Debería ser un Manager.dict
        self.globalNFE = nfe_counter       # Debería ser un Manager.Value

        # Instancia del evaluador (no necesita ser compartido directamente si no tiene estado mutable complejo)
        self.evaluador = evaluadorBlosum()

        # Parámetros de interacción (pueden ser atributos de la clase)
        self.dAttr: float = 0.1
        self.wAttr: float = 0.2
        self.hRep: float = 0.1
        self.wRepel: float = 0.1

    def set_interaction_params(self, dAttr: float, wAttr: float, hRep: float, wRepel: float):
        """Establece los parámetros para las funciones de atracción y repulsión."""
        self.dAttr = dAttr
        self.wAttr = wAttr
        self.hRep = hRep
        self.wRepel = wRepel
        # print(f"Parámetros de interacción actualizados: dAttr={self.dAttr}, wAttr={self.wAttr}, hRep={self.hRep}, wRepel={self.wRepel}") # Debug print

    # Agregado: Método para obtener la lista de fitness
    def get_fitness_list(self) -> List[float]:
        """Retorna una copia local de la lista de fitness."""
        try:
            return list(self.tablaFitness)
        except Exception as e:
            print(f"Error obteniendo la lista de fitness: {e}")
            return []

    # Método para inicializar una bacteria aleatoria (utilizado en E-D)
    def initialize_random_bacteria(self, base_sequences: List[List[str]]) -> Optional[Tuple[Tuple[str, ...], ...]]:
        """
        Inicializa un solo alineamiento aleatorio basado en secuencias base.
        Devuelve una tupla de tuplas para ser compatible con Manager.list.
        """
        if not base_sequences:
            print("Error: No se proporcionaron secuencias base para la inicialización aleatoria.")
            return None

        # Crea una copia profunda mutable (lista de listas)
        new_alignment_mutable: List[List[str]] = [list(seq) for seq in base_sequences]
        num_seqs = len(new_alignment_mutable)
        if num_seqs == 0:
            return tuple() # Devuelve una tupla vacía si no hay secuencias

        # Añadir un número aleatorio de gaps iniciales
        total_base_len = sum(len(seq) for seq in base_sequences)
        # Añade un número aleatorio de gaps, e.g., hasta 10% de la longitud base total
        num_gaps_to_add = random.randint(0, max(1, total_base_len // 10))

        for _ in range(num_gaps_to_add):
            if not new_alignment_mutable: continue # Seguridad
            try:
                # Seleccionar un índice de secuencia válido y no vacío
                valid_seq_indices = [idx for idx, seq in enumerate(new_alignment_mutable) if seq]
                if not valid_seq_indices:
                     # print("Advertencia: Todas las secuencias están vacías durante inserción random de gap.")
                     break # Salir del bucle for _ in range(numGaps_to_add)

                seq_idx = random.choice(valid_seq_indices)
                pos = random.randint(0, len(new_alignment_mutable[seq_idx]))
                new_alignment_mutable[seq_idx].insert(pos, "-")
            except IndexError:
                 print(f"Warning: IndexError durante inserción random de gap.")
                 continue # Continuar intentando con otros gaps
            except Exception as e:
                print(f"Warning: Error inesperado durante inserción random de gap: {e}")
                # Continuar intentando con otros gaps

        # Convertir a tupla de tuplas para almacenamiento en Manager.list
        try:
            return tuple(tuple(seq) for seq in new_alignment_mutable)
        except Exception as e:
            print(f"Error convirtiendo alineamiento mutable a tupla de tuplas en initialize_random_bacteria: {e}")
            return None # Devolver None si falla la conversión


    # El método cuadra recibe una lista de alineamientos y los ajusta a la misma longitud
    def cuadra(self, numSec: int, poblacion: List[List[List[str]]]):
        """Ajusta la longitud de todas las secuencias en cada alineamiento para que sean iguales."""

        if not poblacion:
             # print("Advertencia: Población vacía pasada a cuadra.")
             return # Nada que cuadrar si la población está vacía

        poblacion_mutable = [list(b) if not isinstance(b, list) else b for b in poblacion if b is not None]

        if not poblacion_mutable:
             # print("Advertencia: Población mutable vacía después de filtrar None en cuadra.")
             return # Nada que cuadrar

        maxLen = 0
        # Calcular la longitud máxima en la población mutable
        for bacterTmp in poblacion_mutable:
            if not bacterTmp: continue # Saltar alineamientos vacíos dentro de la lista mutable
            try:
                # max(len(s) for s in bacterTmp if s) encuentra la longitud máxima de secuencias no vacías dentro del alineamiento
                current_max = max((len(s) for s in bacterTmp if s), default=0)
            except Exception as e:
                print(f"Error calculando maxLen en cuadra: {e}")
                current_max = 0
            if current_max > maxLen:
                maxLen = current_max

        if maxLen == 0:
             # print("Advertencia: Longitud máxima es 0 en cuadra. No se insertarán gaps.")
             return # Nada que cuadrar si todas las secuencias tienen longitud 0

        # Rellenar con gaps hasta la longitud máxima
        for i in range(len(poblacion_mutable)):
            bacterTmp = poblacion_mutable[i]
            if not bacterTmp: continue # Saltar alineamientos vacíos

            num_seqs_in_bact = len(bacterTmp)
            for t in range(num_seqs_in_bact):
                 # Asegurarse de que la secuencia es mutable (lista)
                 if not isinstance(bacterTmp[t], list):
                     bacterTmp[t] = list(bacterTmp[t])

                 gap_count = maxLen - len(bacterTmp[t])
                 if gap_count > 0:
                     bacterTmp[t].extend(["-"] * gap_count)
    # El método tumbo realiza una perturbación (inserta gaps) en un alineamiento
    def tumbo(self, numSec: int, alineamiento: List[List[str]], numGaps: int) -> List[List[str]]:
        """Realiza una operación de 'tumbo' (insertar gaps) en un alineamiento."""
        # alineamiento aquí debe ser una copia local mutable (lista de listas de str)
        # No debe ser un elemento de Manager.list directamente.

        if not alineamiento or not alineamiento[0]:
             # print("Advertencia: Alineamiento vacío o inválido pasado a tumbo. Devolviendo copia.")
             return copy.deepcopy(alineamiento) # Devuelve copia incluso si está vacío

        try:
            bacterTmp = [list(seq) for seq in alineamiento] # Asegura que sea mutable
        except TypeError:
            print("Error convirtiendo alineamiento a lista de listas en tumbo. Devolviendo copia original.")
            return copy.deepcopy(alineamiento)
        except Exception as e:
             print(f"Error inesperado al copiar alineamiento en tumbo: {e}. Devolviendo copia original.")
             return copy.deepcopy(alineamiento)


        for _ in range(numGaps):
            if not bacterTmp or not bacterTmp[0]: continue # Evitar operar en alineamientos vacíos o secuencias vacías

            try:
                # Seleccionar un índice de secuencia válido
                valid_seq_indices = [idx for idx, seq in enumerate(bacterTmp) if seq]
                if not valid_seq_indices:
                     # print("Advertencia: Todas las secuencias están vacías en tumbo. No se insertarán gaps.")
                     break # Salir del bucle for _ in range(numGaps)

                seqnum = random.choice(valid_seq_indices)
                pos = random.randint(0, len(bacterTmp[seqnum]))
                bacterTmp[seqnum].insert(pos, "-")
            except IndexError:
                 print(f"Error de índice en tumbo (seqnum={seqnum}, len(bacterTmp)={len(bacterTmp)}). Saltando inserción de gap.")
                 continue # Intentar insertar gap en la siguiente iteración
            except Exception as e:
                 print(f"Error inesperado en tumbo: {e}. Saltando inserción de gap.")
                 continue # Intentar insertar gap en la siguiente iteración

        # Devuelve el alineamiento perturbado como lista de listas
        return bacterTmp # La conversión a tupla de tuplas debe ocurrir fuera si se guarda en Manager.list

    # Método auxiliar para obtener pares de columnas para evaluación BLOSUM
    def _get_pairs_for_eval(self, alignment_list: List[List[str]]) -> List[Tuple[str, str]]:
        """Obtiene pares de caracteres para la evaluación BLOSUM por columna."""
        pares = []
        if not alignment_list or not alignment_list[0]:
            # print("Advertencia: Alineamiento vacío o inválido en _get_pairs_for_eval.")
            return []

        try:
            num_seqs = len(alignment_list)
            # Verificar que todas las secuencias tengan la misma longitud (después de cuadra, deberían)
            len_seqs = [len(s) for s in alignment_list]
            # Filtrar longitudes no válidas (ej: None o no numéricas) antes de all()
            valid_len_seqs = [l for l in len_seqs if isinstance(l, (int, float)) and l >= 0]

            if not valid_len_seqs:
                 # print("Advertencia: No se encontraron longitudes de secuencia válidas en _get_pairs_for_eval.")
                 return []

            first_len = valid_len_seqs[0]
            if not all(l == first_len for l in valid_len_seqs):
                 print(f"Advertencia: Longitudes inconsistentes detectadas en _get_pairs_for_eval. Longitudes: {len_seqs}. Usando la longitud mínima.")
                 len_seq = min(valid_len_seqs)
            else:
                len_seq = first_len

            if len_seq == 0: return []

            for col_idx in range(len_seq):
                columna = []
                for row_idx in range(num_seqs):
                     try:
                          # Asegurarse de no salir del índice si las longitudes son inconsistentes
                          # y que la secuencia no sea None o vacía
                          seq = alignment_list[row_idx]
                          if seq is not None and isinstance(seq, (list, tuple)) and col_idx < len(seq):
                               columna.append(seq[col_idx])
                          else:
                               # Esto no debería pasar si cuadra funciona bien, pero como protección
                               columna.append("-") # Usar gap si el índice está fuera de rango o secuencia inválida
                     except IndexError:
                          print(f"Error de índice recuperando columna {col_idx}, fila {row_idx}. Usando '-' por defecto.")
                          columna.append("-")
                     except Exception as e:
                          print(f"Error inesperado recuperando caracter: {e}. Usando '-' por defecto.")
                          columna.append("-")

                # Generar pares para cada columna
                for i in range(num_seqs):
                    for j in range(i + 1, num_seqs):
                        par = tuple(sorted((columna[i], columna[j]))) # Ordenar para consistencia en el caché
                        pares.append(par)
        except Exception as e:
             print(f"Error inesperado en _get_pairs_for_eval (bucle principal): {e}")
             import traceback
             traceback.print_exc()
             return []
        return pares


    # Método para calcular el score BLOSUM de un alineamiento (sin caché ni NFE)
    def _calculate_blosum_score(self, alignment_hashable: Optional[Tuple[Tuple[str, ...], ...]]) -> float:
        """Calcula el score BLOSUM total para un alineamiento (sin usar caché ni NFE)."""
        if alignment_hashable is None:
             return -float('inf') # O 0.0, dependiendo de cómo quieras penalizar None

        # Convertir la tupla de tuplas de vuelta a lista de listas para el procesamiento interno
        try:
             alignment_list = [list(seq) for seq in alignment_hashable]
        except Exception as e:
             print(f"Error convirtiendo alignment_hashable a lista de listas en _calculate_blosum_score: {e}")
             return -float('inf')


        pares = self._get_pairs_for_eval(alignment_list)
        score = 0.0
        for par in pares:
            try:
                 # Asegurarse de que los elementos del par sean strings
                 char1 = str(par[0]) if par[0] is not None else "-"
                 char2 = str(par[1]) if par[1] is not None else "-"
                 score += self.evaluador.getScore(char1, char2)
            except Exception as e:
                 print(f"Error obteniendo score para par {par}: {e}")
                 score += self.evaluador.getScore("-", "-") # Penalización por defecto o manejo de error


        return score

    # Método para evaluar la población completa usando el caché BLOSUM
    def evaluaBlosumConCache(self, poblacion: List):
        """Evalúa el score BLOSUM para cada bacteria en la población COMPLETA usando caché."""
        # poblacion aquí debe ser el Manager.list compartido (tupla de tuplas o None)
        if len(poblacion) != self.numBacterias:
             print(f"Advertencia: evaluaBlosumConCache llamada con lista de tamaño {len(poblacion)} en lugar de {self.numBacterias}.")
             # Intentar procesar lo que se pasó, pero no actualizar self.blosumScore si el tamaño no coincide.
             scores = [0.0] * len(poblacion)
             should_update_blosumscore = False
        else:
            scores = [0.0] * self.numBacterias
            should_update_blosumscore = True

        for i in range(len(scores)): # Iterar sobre el tamaño de la lista pasada
            try:
                # Asegurarse de que el alineamiento esté en formato hashable y no sea None
                alignment = poblacion[i]
                if alignment is None:
                     # print(f"Advertencia: Alineamiento es None para bacteria {i}. Usando score 0.")
                     scores[i] = 0.0
                     continue

                alignment_hashable = alignment_to_hashable(alignment)

                if alignment_hashable is None or not alignment_hashable: # Modificado para verificar alignment_hashable is None
                    # print(f"Advertencia: Alineamiento inválido o vacío para bacteria {i}. Usando score 0.")
                    scores[i] = 0.0
                    continue
            except Exception as e:
                 print(f"Error accediendo a la bacteria {i} en evaluaBlosumConCache: {e}. Usando score 0.")
                 import traceback
                 traceback.print_exc()
                 scores[i] = 0.0
                 continue

            try:
                if alignment_hashable in self.fitness_cache:
                    scores[i] = self.fitness_cache[alignment_hashable]
                    # print(f"  Cache hit para bacteria {i}") # Debug print
                else:
                    # print(f"  Cache miss para bacteria {i}. Calculando...") # Debug print
                    score = self._calculate_blosum_score(alignment_hashable)
                    # Incrementar NFE global (sin usar get_lock())
                    # Asegurarse de que globalNFE sea válido antes de incrementar
                    if self.globalNFE is not None:
                         self.globalNFE.value += 1
                    else:
                         print("Advertencia: globalNFE_shared es None al intentar incrementar NFE.")

                    self.fitness_cache[alignment_hashable] = score
                    scores[i] = score
            except TypeError as e:
                 print(f"Error de tipo (posiblemente no hashable) con alineamiento para bacteria {i}: {e}")
                 import traceback
                 traceback.print_exc()
                 # Intentar calcular sin caché si hay un problema de hashable
                 # Convertir de nuevo a hashable por si el error fue temporal, pero usar _calculate_blosum_score
                 scores[i] = self._calculate_blosum_score(alignment_to_hashable(alignment))
            except Exception as e:
                 print(f"Error inesperado durante cache/cálculo para bacteria {i}: {e}")
                 import traceback
                 traceback.print_exc()
                 # Fallback a score 0 o un valor de penalización
                 scores[i] = 0.0

        # Solo actualizar self.blosumScore si se evaluó la población completa
        if should_update_blosumscore:
            try:
                self.blosumScore[:] = scores
            except Exception as e:
                 print(f"Error actualizando self.blosumScore (población completa): {e}")
                 import traceback
                 traceback.print_exc()
                 # Considerar si es un error fatal o si se puede continuar con valores anteriores


    # Nuevo método para evaluar UN SOLO alineamiento usando caché/NFE
    def evaluaSingleBlosumWithCache(self, alignment_tuple: Optional[Tuple[Tuple[str, ...], ...]]) -> float:
        """Evalúa el score BLOSUM de un solo alineamiento usando caché y NFE."""
        score = 0.0
        try:
            if alignment_tuple is None:
                 # print("Advertencia: Alineamiento individual es None. Usando score 0.")
                 return -float('inf') # Penalizar alineamientos None

            alignment_hashable = alignment_to_hashable(alignment_tuple)

            if alignment_hashable is None or not alignment_hashable: # Modificado para verificar alignment_hashable is None
                 # print("Advertencia: Alineamiento individual inválido o vacío. Usando score 0.")
                 return -float('inf') # Penalizar alineamientos inválidos

            if alignment_hashable in self.fitness_cache:
                score = self.fitness_cache[alignment_hashable]
                # print("  Cache hit para alineamiento individual") # Debug print
            else:
                # print("  Cache miss para alineamiento individual. Calculando...") # Debug print
                score = self._calculate_blosum_score(alignment_hashable)
                # Incrementar NFE global (sin usar get_lock())
                if self.globalNFE is not None:
                     self.globalNFE.value += 1
                else:
                    print("Advertencia: globalNFE_shared es None al intentar incrementar NFE en evaluaSingleBlosumWithCache.")

                self.fitness_cache[alignment_hashable] = score
        except TypeError as e:
            print(f"Error de tipo (posiblemente no hashable) con alineamiento individual: {e}")
            import traceback
            traceback.print_exc()
            # Fallback a cálculo sin caché
            score = self._calculate_blosum_score(alignment_to_hashable(alignment_tuple))
        except Exception as e:
            print(f"Error inesperado durante cache/cálculo para alineamiento individual: {e}")
            import traceback
            traceback.print_exc()
            score = -float('inf') # Penalizar errores inesperados

        return score


    # Método inicial de evaluación (llama al método con caché para la población completa)
    def evaluaBlosumInicial(self, poblacion: List):
        """Realiza la evaluación inicial de la población."""
        print("  (Llamando a evaluaBlosumConCache para evaluación inicial)")
        self.evaluaBlosumConCache(poblacion)

    # --- Métodos de Interacción ---

    # Función auxiliar para calcular el término exponencial de la interacción para UN par de scores BLOSUM
    # Ajustado para devolver el término SIN el signo de atracción/repulsión aún
    def compute_interaction_term(self, score_self: float, score_other: float, w: float) -> float:
        """Calcula el término exponencial exp(-w * diff^2) para un par de scores."""
        try:
            # Asegurarse de que los scores sean finitos
            if not np.isfinite(score_self) or not np.isfinite(score_other):
                 diff = float('inf')
            else:
                 diff = (score_self - score_other) ** 2.0

            if w < 0:
                 # print(f"Advertencia: 'w' negativo ({w}) pasado a compute_interaction_term. Usando valor absoluto.")
                 w = abs(w)
            if w == 0:
                 # Si w es cero, exp(0)=1. El término es 1 antes de multiplicar por d/h.
                 return 1.0

            # Calcular el término exponencial
            if diff == float('inf'):
                return 0.0 # Diferencia infinita => término exponencial es 0
            elif diff == 0:
                 return 1.0 # Si los scores son iguales, exp(0)=1
            else:
                exponent = -w * diff
                # Evitar overflow/underflow con grandes exponentes
                if exponent < -700: # approx. log(0) for float64
                     return 0.0
                elif exponent > 700: # approx. log(inf) for float64
                     # Esto no debería ocurrir con -w * diff (exponente negativo)
                     # Si ocurre, indica un problema lógico.
                     print(f"Advertencia: Exponente positivo inesperado ({exponent}) en compute_interaction_term.")
                     return float('inf') # O 0.0 para seguridad

                term = np.exp(exponent)

            # Asegurarse de que el resultado sea finito y no NaN
            if not np.isfinite(term):
                 # print(f"Advertencia: Valor de término de interacción no finito ({term}) para scores ({score_self}, {score_other}, w={w}). Usando 0.0.")
                 return 0.0

        except OverflowError:
             print(f"Error de overflow en compute_interaction_term (scores={score_self}, {score_other}, w={w}). Usando 0.0.")
             return 0.0
        except Exception as e:
             print(f"Error inesperado en compute_interaction_term para scores ({score_self}, {score_other}, w={w}): {e}")
             import traceback
             traceback.print_exc()
             return 0.0 # Manejo general de errores

        return term

    def compute_cell_interaction(self, indexBacteria: int):
        """
        Calcula el término de interacción total (atracción + repulsión) para una bacteria específica
        con respecto a toda la población. Cálculo serial.
        """

        total_attract_term = 0.0
        total_repel_term = 0.0

        try:
             # Obtener copias locales de las listas compartidas
             allBlosumScores_local = list(self.blosumScore)
             num_bacterias = len(allBlosumScores_local)

             # Asegurarse de que el índice de la bacteria actual sea válido y su score finito
             if indexBacteria < 0 or indexBacteria >= num_bacterias or not np.isfinite(allBlosumScores_local[indexBacteria]):
                  # print(f"Advertencia: Datos inválidos para calcular interacción para bacteria {indexBacteria}. Num bacterias: {num_bacterias}")
                  # Asegurarse de que las tablas se inicialicen a 0.0 si este índice es válido en numBacterias total
                  if indexBacteria < self.numBacterias:
                       self.tablaAtract[indexBacteria] = 0.0
                       self.tablaRepel[indexBacteria] = 0.0
                  return # No hay nada que calcular si la bacteria o su score son inválidos

             score_self = allBlosumScores_local[indexBacteria]

             # Calcular la interacción con cada OTRA bacteria en la población
             for j in range(num_bacterias):
                 if indexBacteria == j: continue # Excluir interacción consigo mismo

                 other_blosum_score = allBlosumScores_local[j]

                 # Asegurarse de que el score del otro sea finito
                 if not np.isfinite(other_blosum_score):
                      # print(f"Advertencia: Score de bacteria {j} no finito ({other_blosum_score}) al calcular interacción con bacteria {indexBacteria}. Saltando.")
                      continue # Saltar interacción con esta bacteria si su score no es finito


                 # Calcular el término de interacción (exponencial) basado en la diferencia de scores
                 term_exp_attr = self.compute_interaction_term(score_self, other_blosum_score, self.wAttr)
                 total_attract_term += term_exp_attr

                 term_exp_repel = self.compute_interaction_term(score_self, other_blosum_score, self.wRepel)
                 total_repel_term += term_exp_repel


             # Aplicar d/h y signo negativo a los sumatorios totales ANTES de almacenar
             # La tabla de atracción almacena Sum(-d_attr * exp(...))
             self.tablaAtract[indexBacteria] = -self.dAttr * total_attract_term if np.isfinite(total_attract_term) else 0.0
             # La tabla de repulsión almacena Sum(h_repel * exp(...))
             self.tablaRepel[indexBacteria] = self.hRep * total_repel_term if np.isfinite(total_repel_term) else 0.0


        except IndexError:
             print(f"Error de índice en compute_cell_interaction para bacteria {indexBacteria}.")
             import traceback
             traceback.print_exc()
             if indexBacteria < self.numBacterias:
                  self.tablaAtract[indexBacteria] = 0.0
                  self.tablaRepel[indexBacteria] = 0.0
        except Exception as e:
             print(f"Error inesperado en compute_cell_interaction para bacteria {indexBacteria}: {e}")
             import traceback
             traceback.print_exc()
             if indexBacteria < self.numBacterias:
                  self.tablaAtract[indexBacteria] = 0.0
                  self.tablaRepel[indexBacteria] = 0.0


    # Métodos para crear las tablas de atracción y repulsión llamando a compute_cell_interaction (Serial)
    def creaTablaAtract(self, poblacion: List): # poblacion no se usa aquí, compute_cell_interaction uses self.blosumScore
        """Calcula y llena la tabla de atracción para toda la población."""
     
        try:
            # Iterate through all bacteria by their index
            num_bacterias = self.numBacterias # Use the expected population size
            for indexBacteria in range(num_bacterias):
                 # compute_cell_interaction now calculates BOTH tables (atracción and repulsión) for an index
                 self.compute_cell_interaction(indexBacteria) # Call only once per index
            # print("    Attraction/Repulsion calculations completed (via compute_cell_interaction).") # Debug print
        except Exception as e:
             print(f"Error durante la creación de la tabla de Atracción/Repulsión: {e}")
             import traceback
             traceback.print_exc()
             # The values in tablaAtract/tablaRepel might be incorrect/incomplete

    def creaTablaRepel(self, poblacion: List): # This method is no longer needed if creaTablaAtract calls compute_cell_interaction
         """Calcula y llena la tabla de repulsión para toda la población."""

         pass # The logic is already in creaTablaAtract if it calls compute_cell_interaction for each index


    # Método combinado para crear tablas de atracción y repulsión
    def creaTablasAtractRepel(self, poblacion: List):
         """Calcula y llena las tablas de atracción y repulsión."""
 
         try:
             # Call creaTablaAtract which in turn calls compute_cell_interaction for each bacterium.
             self.creaTablaAtract(poblacion)
            
         except Exception as e:
             print(f"Error in creaTablasAtractRepel: {e}")
             import traceback
             traceback.print_exc()


    # Método para crear la tabla de interacción total (Atracción + Repulsión)
    def creaTablaInteraction(self, poblacion: List): # poblacion no se usa aquí
        """Calcula la tabla de interacción total (atracción + repulsión) para cada bacteria."""


        try:
            len_atract = len(self.tablaAtract)
            len_repel = len(self.tablaRepel)
            target_len = self.numBacterias # Use the expected population size

             # Ensure the shared lists have the expected size before iterating
            if len_atract != target_len or len_repel != target_len:
                 print(f"Warning: Sizes of attraction/repulsion tables ({len_atract}, {len_repel}) do not match numBacterias ({target_len}) in creaTablaInteraction.")
                 # Fill interaction_results with 0s if there's a size issue
                 interaction_results = [0.0] * target_len
            else:
                 interaction_results = [0.0] * target_len
                 # Sum the attraction and repulsion terms summed separately
                 for i in range(target_len):
                     # Access shared lists safely (we already checked size above)
                     atract_sum = self.tablaAtract[i]
                     repel_sum = self.tablaRepel[i]

                     sum_interact = atract_sum + repel_sum
                     if not np.isfinite(sum_interact):
                          # print(f"Warning: Non-finite interaction sum for bacterium {i}. Using 0.0.")
                          sum_interact = 0.0
                     interaction_results[i] = sum_interact

            # Update the shared list of total interaction (Sum(Attraction) + Sum(Repulsion))
            self.tablaInteraction[:] = interaction_results
            # print("    Creation of Interaction table completed.") # Debug print
        except Exception as e:
            print(f"Error in creaTablaInteraction: {e}")
            import traceback
            traceback.print_exc()
            # Reset tablaInteraction or handle the error otherwise
            self.tablaInteraction[:] = [0.0] * self.numBacterias


    # Método para crear la tabla de fitness total (BLOSUM + Interacción)
    def creaTablaFitness(self, poblacion: List): # poblacion is not used here
        """Calculates the total fitness (BLOSUM score + interaction) for each bacterium."""

        try:
            len_blosum = len(self.blosumScore)
            len_interact = len(self.tablaInteraction)
            target_len = self.numBacterias # Use the expected population size

             # Ensure the shared lists have the expected size before iterating
            if len_blosum != target_len or len_interact != target_len:
                 print(f"Warning: Sizes of blosum/interaction tables ({len_blosum}, {len_interact}) do not match numBacterias ({target_len}) in creaTablaFitness.")
                 # Fill fitness_results with -inf if there's a size issue
                 fitness_results = [-float('inf')] * target_len
            else:
                 fitness_results = [0.0] * target_len
                 # Sum BLOSUM score and total interaction score (Sum(Attr) + Sum(Repel))
                 for i in range(target_len):
                     # Access shared lists safely (we already checked size above)
                     valorBlsm = self.blosumScore[i]
                     valorInteract = self.tablaInteraction[i]

        
                     if not np.isfinite(valorBlsm):
                          valorBlsm = -float('inf') # Penalize heavily if the base score is not finite

                     if not np.isfinite(valorInteract):
                          valorInteract = 0.0 # Non-finite interaction can be considered 0 or handled per theory.

                     fitness_results[i] = valorBlsm + valorInteract # Maximize fitness

            # Update the shared list of total fitness
            self.tablaFitness[:] = fitness_results
            # print("    Creation of Fitness table completed.") # Debug print
        except Exception as e:
             print(f"Error in creaTablaFitness: {e}")
             import traceback
             traceback.print_exc()
             # Reset tablaFitness to a low value to indicate failure
             self.tablaFitness[:] = [-float('inf')] * self.numBacterias # Use -inf if maximizing

    # Method to get the global NFE value
    def getNFE(self) -> int:
        """Returns the accumulated number of function evaluations (NFE)."""
        try:
            # Access Manager.Value directly is safe for reading
            if self.globalNFE is not None:
                 return self.globalNFE.value
            else:
                 # print("Warning: globalNFE_shared is None when trying to get NFE.")
                 return -1
        except Exception as e:
             print(f"Error accessing global NFE: {e}")
             import traceback
             traceback.print_exc()
             return -1 # Return -1 or some indicator of error

    # Method to find the best bacterium (based on fitness)
    def obtieneBest(self) -> Tuple[int, float]:
        """Finds the index and fitness of the best bacterium in the current population."""
        bestIdx = -1
        best_fitness_val = -float('inf') # Initialize with minimum value if maximizing

        try:
            fitness_list_local = list(self.tablaFitness) # Get a local copy for safe iteration
            num_bacterias_actual = len(fitness_list_local)

            if num_bacterias_actual == 0:
                 # print("Warning: tablaFitness is empty in obtieneBest.")
                 return -1, -float('inf')

            # Find the first index with finite fitness as a starting point
            valid_indices = [i for i, f in enumerate(fitness_list_local) if np.isfinite(f)]
            if not valid_indices:
                 # print("Warning: No valid finite fitness found in obtieneBest.")
                 # If no finite fitnesses, return the first one (which will be -inf) or an indicator.
                 # Return index 0 and its value (which will be -inf if no finite) as a fallback.
                 return 0, fitness_list_local[0] if fitness_list_local else -float('inf')


            bestIdx = valid_indices[0]
            best_fitness_val = fitness_list_local[bestIdx]

            # Iterate over the rest to find the maximum
            for i in valid_indices:
                 current_val = fitness_list_local[i]
                 if current_val > best_fitness_val: # Maximize
                     best_fitness_val = current_val
                     bestIdx = i


        except Exception as e:
             print(f"Error in obtieneBest: {e}")
             import traceback
             traceback.print_exc()
             return -1, -float('inf') # Return error indicator

        return bestIdx, best_fitness_val

    # Method to replace the worst bacterium with the best (in reproduction)
    def replaceWorst(self, poblacion: List, bestIdx: int):
        """Replaces the bacterium with the worst fitness with a copy of the best one."""
        # poblacion is the shared Manager.list (tuple of tuples or None)
        worstIdx = -1
        worst_fitness_val = float('inf') # Initialize with maximum value if maximizing

        try:
            fitness_list_local = list(self.tablaFitness) # Local copy
            num_bacterias_actual = len(fitness_list_local) # Actual size of the fitness list

            # Validate bestIdx and the size of the shared population
            if bestIdx < 0 or bestIdx >= len(poblacion) or len(poblacion) != self.numBacterias:
                 print("Warning: Cannot execute replaceWorst (bestIdx out of range or shared population has incorrect size).")
                 return

            # Find the first index with finite fitness as a starting point for the worst
            valid_indices = [i for i, f in enumerate(fitness_list_local) if np.isfinite(f)]
            if not valid_indices:
                 print("Warning: No valid finite fitness found to determine the worst.")
                 return

            worstIdx = valid_indices[0]
            worst_fitness_val = fitness_list_local[worstIdx]

            # Iterate to find the minimum finite fitness
            for i in valid_indices:
                 current_val = fitness_list_local[i]
                 if current_val < worst_fitness_val: # Minimize (find the worst)
                     worst_fitness_val = current_val
                     worstIdx = i

            # Perform the replacement if a valid worst index different from the best was found
            # Ensure worstIdx is within the range of the shared population
            if worstIdx != -1 and worstIdx != bestIdx and 0 <= worstIdx < len(poblacion) and poblacion[bestIdx] is not None:
                # Perform a deep copy of the best bacterium (which is a tuple of tuples)
                poblacion[worstIdx] = copy.deepcopy(poblacion[bestIdx])

                # Copy the corresponding scores and table values as well
                # Accessing shared lists (ensure indices are valid)
                if worstIdx < len(self.blosumScore): self.blosumScore[worstIdx] = self.blosumScore[bestIdx]
                if worstIdx < len(self.tablaFitness): self.tablaFitness[worstIdx] = self.tablaFitness[bestIdx]
                if worstIdx < len(self.tablaInteraction): self.tablaInteraction[worstIdx] = self.tablaInteraction[bestIdx]
                if worstIdx < len(self.tablaAtract): self.tablaAtract[worstIdx] = self.tablaAtract[bestIdx]
                if worstIdx < len(self.tablaRepel): self.tablaRepel[worstIdx] = self.tablaRepel[bestIdx]

                # print(f"    Replacing worst bacterium (Index {worstIdx}, Fitness {worst_fitness_val:.4f}) with copy of the best (Index {bestIdx}).")
            elif worstIdx == bestIdx:
                 # print(f"    The best bacterium (Index {bestIdx}) is also the worst or is the only valid one. No replacement is performed.")
                 pass # Do nothing if the best is the worst
            elif worstIdx != -1 and 0 <= worstIdx < len(poblacion):
                 print(f"Warning: The best bacterium (Index {bestIdx}) is None. Cannot replace the worst (Index {worstIdx}).")
            else:
                 print(f"Warning: Could not find a valid worst bacterium to replace (worstIdx={worstIdx}, bestIdx={bestIdx}).")

        except IndexError:
             print(f"Error de índice during replaceWorst (accessing poblacion or tables).")
             import traceback
             traceback.print_exc()
        except Exception as e:
             print(f"Unexpected error in replaceWorst: {e}")
             import traceback
             traceback.print_exc() # Print traceback for debugging


    # Method to get the best alignment found
    def get_best_alignment(self, bestIdx: int, poblacion: List) -> Optional[List[str]]:
        """Retrieves the alignment of the best bacterium and formats it as a list of strings."""
        # poblacion is the shared Manager.list (tuple of tuples or None)

        # Check if the index is valid and the alignment is not None
        if 0 <= bestIdx < len(poblacion):
            alignment_tuple: Optional[Tuple[Tuple[str, ...], ...]] = poblacion[bestIdx]
            if alignment_tuple is None:
                 print(f"Warning: Alignment for index {bestIdx} is None.")
                 return None # Return None if the alignment is None
            try:
                # The alignment is stored as a tuple of tuple of characters
                # Convert to list of strings for output/display
                alignment_strings = ["".join(seq) for seq in alignment_tuple]
                return alignment_strings
            except Exception as e:
                print(f"Error retrieving/formatting best alignment for index {bestIdx}: {e}")
                import traceback
                traceback.print_exc()
                return None # Return None if there's an error
        else:
            print(f"Error: Best index ({bestIdx}) out of range ({len(poblacion)}) for the population.")
            return None # Return None if the index is not valid


    # --- New method to calculate total fitness (Blosum + Interaction) for an alignment ---
    # Needs the BLOSUM scores of the current population to calculate interaction.
    def calculate_total_fitness_for_alignment(self, external_alignment_tuple: Optional[Tuple[Tuple[str, ...], ...]], current_population_blosum_scores: List[float]) -> float:
        """
        Calculates the total fitness (Blosum + Interaction) for an external alignment
        in the context of the BLOSUM scores of the current population.
        This DOES NOT modify the bacterium's internal tables or increment NFE (except via evaluaSingleBlosumWithCache).
        """
        if external_alignment_tuple is None:
            return -float('inf') # Penalize if alignment is None

        # 1. Calculate the BLOSUM score of the external alignment (uses cache/NFE)
        external_blosum_score = self.evaluaSingleBlosumWithCache(external_alignment_tuple)

        if not np.isfinite(external_blosum_score):
             # If the Blosum score is not finite, the total fitness is also non-finite or -inf.
             return -float('inf')

        # 2. Calculate the interaction of this external alignment with the current population
        total_attract_term_external_sum = 0.0
        total_repel_term_external_sum = 0.0

        try:
            num_bacterias = len(current_population_blosum_scores)

            # Interaction with each bacterium in the population
            for j in range(num_bacterias):
                other_blosum_score = current_population_blosum_scores[j]

                # Ensure the other's score is finite
                if not np.isfinite(other_blosum_score):
                     continue # Skip interaction with this bacterium if its score is not finite


                # Calculate the interaction term (exponential) based on the score difference
                term_exp_attr = self.compute_interaction_term(external_blosum_score, other_blosum_score, self.wAttr)
                total_attract_term_external_sum += term_exp_attr

                term_exp_repel = self.compute_interaction_term(external_blosum_score, other_blosum_score, self.wRepel)
                total_repel_term_external_sum += term_exp_repel


            # Apply d/h and signs to the total sums for the external alignment
            # The Attraction term for the external alignment is Sum_j (-d_attr * exp(-w_attr * diff^2))
            total_attract_for_external = -self.dAttr * total_attract_term_external_sum if np.isfinite(total_attract_term_external_sum) else 0.0
            # The Repulsion term for the external alignment is Sum_j (h_repel * exp(-w_repel * diff^2))
            total_repel_for_external = self.hRep * total_repel_term_external_sum if np.isfinite(total_repel_term_external_sum) else 0.0

            # The total interaction for the external alignment is the sum of total attraction and repulsion
            interaction_value = total_attract_for_external + total_repel_for_external

            # Ensure the calculated interaction is finite
            if not np.isfinite(interaction_value):
                 interaction_value = 0.0


        except Exception as e:
             print(f"Error calculating interaction for external alignment: {e}")
             import traceback
             traceback.print_exc()
             interaction_value = 0.0 # Default to 0 interaction on error

        # Total fitness is Blosum score + Interaction score
        total_fitness = external_blosum_score + interaction_value

        # Ensure total fitness is finite
        if not np.isfinite(total_fitness):
             total_fitness = -float('inf') # Set to -inf if result is not finite

        return total_fitness