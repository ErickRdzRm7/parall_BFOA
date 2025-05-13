import copy
import math
import random
import numpy as np
# Eliminar Pool y Lock ya que no se usarán en compute_cell_interaction
from multiprocessing import Manager, Value # Mantener Manager y Value
from typing import List, Tuple, Dict, Any, Optional

# Asegúrate de que evaluadorBlosum esté en el mismo directorio o accesible en el PATH
from evaluadorBlosum import evaluadorBlosum

# --- Función auxiliar para convertir alineamiento a hashable ---
# Se mantiene fuera de la clase para serialización en multiprocessing si es necesario,
# aunque la conversión a tuple() ya lo hace hashable si los elementos son hashable.
# Usamos tuple de tuples de str para la representación compartida.
def alignment_to_hashable(alignment):
    if isinstance(alignment, tuple) and all(isinstance(seq, tuple) for seq in alignment):
        return alignment # Ya está en formato hashable (tupla de tuplas)
    # Convertir lista de listas de chars a tupla de tuplas de chars
    # Manejar el caso de alineamiento None o vacío
    if alignment is None:
        return None
    try:
        return tuple(tuple(seq) for seq in alignment)
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
                seq_idx = random.randint(0, num_seqs - 1)
                # Asegurarse de que la secuencia no esté vacía después de alguna operación de tumbo previa
                if not new_alignment_mutable[seq_idx]:
                    continue
                pos = random.randint(0, len(new_alignment_mutable[seq_idx]))
                new_alignment_mutable[seq_idx].insert(pos, "-")
            except IndexError:
                 print(f"Warning: IndexError durante inserción de gap aleatorio en initialize_random_bacteria.")
                 continue # Continuar intentando con otros gaps
            except Exception as e:
                print(f"Warning: Error inesperado durante inserción de gap aleatorio en initialize_random_bacteria: {e}")
                # Continuar intentando con otros gaps

        # Convertir a tupla de tuplas para almacenamiento en Manager.list
        return tuple(tuple(seq) for seq in new_alignment_mutable)


    # El método cuadra recibe una lista de alineamientos y los ajusta a la misma longitud
    def cuadra(self, numSec: int, poblacion: List[List[List[str]]]):
        """Ajusta la longitud de todas las secuencias en cada alineamiento para que sean iguales."""
        # Nota: poblacion aquí debe ser una lista local de alineamientos mutables (lista de listas de str),
        # no el Manager.list compartido directamente, para permitir modificaciones in-place.
        # La conversión de Manager.list a list debe ocurrir antes de llamar a cuadra
        # y la asignación de vuelta a Manager.list debe ocurrir después.

        if not poblacion:
             # print("Advertencia: Población vacía pasada a cuadra.")
             return # Nada que cuadrar si la población está vacía

        # Asegúrate de que la entrada sea mutable si no lo es ya
        # Filter None values if they exist, to avoid errors
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

        # Nota: Las modificaciones se realizaron en poblacion_mutable.
        # El código que llama a cuadra debe manejar la asignación de vuelta a la estructura compartida.
        # No asignamos de vuelta a Manager.list aquí dentro, solo modificamos la lista pasada como argumento.


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
            if not all(l == len_seqs[0] for l in len_seqs):
                 print(f"Advertencia: Longitudes inconsistentes detectadas en _get_pairs_for_eval. Longitudes: {len_seqs}. Usando la longitud mínima.")
                 len_seq = min(len_seqs) if len_seqs else 0
            else:
                len_seq = len_seqs[0] if len_seqs else 0

            if len_seq == 0: return []

            for col_idx in range(len_seq):
                columna = []
                for row_idx in range(num_seqs):
                     try:
                          # Asegurarse de no salir del índice si las longitudes son inconsistentes
                          if col_idx < len(alignment_list[row_idx]):
                               columna.append(alignment_list[row_idx][col_idx])
                          else:
                               # Esto no debería pasar si cuadra funciona bien, pero como protección
                               columna.append("-") # Usar gap si el índice está fuera de rango
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
             return []
        return pares


    # Método para calcular el score BLOSUM de un alineamiento (sin caché ni NFE)
    def _calculate_blosum_score(self, alignment_hashable: Tuple[Tuple[str, ...], ...]) -> float:
        """Calcula el score BLOSUM total para un alineamiento (sin usar caché ni NFE)."""
        # Convertir la tupla de tuplas de vuelta a lista de listas para el procesamiento interno
        alignment_list = [list(seq) for seq in alignment_hashable]
        pares = self._get_pairs_for_eval(alignment_list)
        score = 0.0
        for par in pares:
            try:
                 score += self.evaluador.getScore(par[0], par[1])
            except Exception as e:
                 print(f"Error obteniendo score para par {par}: {e}")
                 score += self.evaluador.getScore("-", "-") # Penalización por defecto o manejo de error

        # Añadir penalización por gaps totales si se desea
        # total_gaps = sum(seq.count('-') for seq in alignment_hashable)
        # gap_penalty = total_gaps * penalizacion_por_gap_individual # Define penalizacion_por_gap_individual
        # score -= gap_penalty

        return score

    # Método para evaluar la población completa usando el caché BLOSUM
    def evaluaBlosumConCache(self, poblacion: List):
        """Evalúa el score BLOSUM para cada bacteria en la población COMPLETA usando caché."""
        # poblacion aquí debe ser el Manager.list compartido (tupla de tuplas)
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

                if alignment_hashable is None or not alignment_hashable or not all(alignment_hashable):
                    # print(f"Advertencia: Alineamiento inválido o vacío para bacteria {i}. Usando score 0.")
                    scores[i] = 0.0
                    continue
            except Exception as e:
                 print(f"Error accediendo a la bacteria {i} en evaluaBlosumConCache: {e}. Usando score 0.")
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
                    self.globalNFE.value += 1
                    self.fitness_cache[alignment_hashable] = score
                    scores[i] = score
            except TypeError as e:
                 print(f"Error de tipo (posiblemente no hashable) con alineamiento para bacteria {i}: {e}")
                 # Intentar calcular sin caché si hay un problema de hashable
                 scores[i] = self._calculate_blosum_score(alignment_to_hashable(alignment)) # Recalcular por si acaso
            except Exception as e:
                 print(f"Error inesperado durante cache/cálculo para bacteria {i}: {e}")
                 # Fallback a score 0 o un valor de penalización
                 scores[i] = 0.0

        # Solo actualizar self.blosumScore si se evaluó la población completa
        if should_update_blosumscore:
            try:
                self.blosumScore[:] = scores
            except Exception as e:
                 print(f"Error actualizando self.blosumScore (población completa): {e}")
                 # Considerar si es un error fatal o si se puede continuar con valores anteriores


    # Nuevo método para evaluar UN SOLO alineamiento usando caché/NFE
    def evaluaSingleBlosumWithCache(self, alignment_tuple: Optional[Tuple[Tuple[str, ...], ...]]) -> float:
        """Evalúa el score BLOSUM de un solo alineamiento usando caché y NFE."""
        score = 0.0
        try:
            if alignment_tuple is None:
                 # print("Advertencia: Alineamiento individual es None. Usando score 0.")
                 return 0.0

            alignment_hashable = alignment_to_hashable(alignment_tuple)

            if alignment_hashable is None or not alignment_hashable or not all(alignment_hashable):
                 # print("Advertencia: Alineamiento individual inválido o vacío. Usando score 0.")
                 return 0.0

            if alignment_hashable in self.fitness_cache:
                score = self.fitness_cache[alignment_hashable]
                # print("  Cache hit para alineamiento individual") # Debug print
            else:
                # print("  Cache miss para alineamiento individual. Calculando...") # Debug print
                score = self._calculate_blosum_score(alignment_hashable)
                # Incrementar NFE global (sin usar get_lock())
                self.globalNFE.value += 1
                self.fitness_cache[alignment_hashable] = score
        except TypeError as e:
            print(f"Error de tipo (posiblemente no hashable) con alineamiento individual: {e}")
            # Fallback a cálculo sin caché
            score = self._calculate_blosum_score(alignment_to_hashable(alignment_tuple))
        except Exception as e:
            print(f"Error inesperado durante cache/cálculo para alineamiento individual: {e}")
            score = 0.0

        return score


    # Método inicial de evaluación (llama al método con caché para la población completa)
    def evaluaBlosumInicial(self, poblacion: List):
        """Realiza la evaluación inicial de la población."""
        print("  (Llamando a evaluaBlosumConCache para evaluación inicial)")
        self.evaluaBlosumConCache(poblacion)

    # --- Métodos de Interacción ---

    # Función auxiliar para calcular la diferencia al cuadrado para la interacción
    def compute_diff(self, args: Tuple[int, float, List[float], float, float, float, float]) -> float:
        """Calcula el término exponencial de la interacción para un par de bacterias."""
        indexBacteria, otherBlosumScore, allBlosumScores, d, w, otherFitness, currentFitness = args
        interaction_value = 0.0
        try:
            score_self = allBlosumScores[indexBacteria]
            score_other = otherBlosumScore

            # Manejar casos donde el score es inválido/None/infinito
            if not np.isfinite(score_self) or not np.isfinite(score_other):
                 diff = float('inf')
            else:
                 diff = (score_self - score_other) ** 2.0

            # Manejo de parámetros de interacción
            if w < 0:
                 # print(f"Advertencia: 'w' negativo ({w}) pasado a compute_diff. Usando valor absoluto.")
                 w = abs(w)
            # Si w es cero, el exponente es 0, exp(0)=1. La interacción es solo 'd'.
            if w == 0:
                 interaction_value = d
                 return interaction_value

            # Calcular el término exponencial
            if diff == float('inf'):
                interaction_value = 0.0 # Diferencia infinita => no interacción por esta diferencia
            elif diff == 0:
                 interaction_value = d # Si los scores son iguales, exp(0)=1, término es 'd'
            else:
                exponent = -w * diff
                # Evitar overflow/underflow con grandes exponentes
                if exponent < -700: # approx. log(0) for float64
                     interaction_value = 0.0
                elif exponent > 700: # approx. log(inf) for float64
                     interaction_value = float('inf') # Esto no debería ocurrir con -w * diff
                else:
                    interaction_value = d * np.exp(exponent)

            # Asegurarse de que el resultado sea finito
            if not np.isfinite(interaction_value):
                 # print(f"Advertencia: Valor de interacción no finito ({interaction_value}) para args {args}. Usando 0.0.")
                 interaction_value = 0.0
        except IndexError:
             print(f"Error de índice en compute_diff accediendo a scores. Usando interacción 0.")
             interaction_value = 0.0
        except OverflowError:
             print(f"Error de overflow en compute_diff (d={d}, w={w}, diff={diff}). Usando 0.")
             interaction_value = 0.0
        except Exception as e:
             print(f"Error inesperado en compute_diff para args {args}: {e}")
             interaction_value = 0.0 # Manejo general de errores

        return interaction_value


    # Método para calcular la interacción de una célula con el resto de la población (Serial)
    def compute_cell_interaction(self, indexBacteria: int, d: float, w: float, atracTrue: bool):
        """
        Calcula el término de interacción (atracción o repulsión) para una bacteria específica
        con respecto a toda la población. Cálculo serial.
        """
        total = 0.0
        try:
            # Obtener copias locales de las listas compartidas
            allBlosumScores_local = list(self.blosumScore)
            allFitnessScores_local = list(self.tablaFitness)

            num_bacterias = len(allBlosumScores_local)
            if num_bacterias == 0 or indexBacteria >= num_bacterias or not np.isfinite(allBlosumScores_local[indexBacteria]):
                 # print(f"Advertencia: Datos inválidos para calcular interacción para bacteria {indexBacteria}. Num bacterias: {num_bacterias}")
                 if indexBacteria < self.numBacterias: # Asegurarse de no causar IndexError al inicializar la tabla
                     if atracTrue:
                         self.tablaAtract[indexBacteria] = 0.0
                     else:
                         self.tablaRepel[indexBacteria] = 0.0
                 return # No hay nada que calcular si no hay bacterias o el score propio es inválido

            # Prepara los argumentos para compute_diff para cada par (bacteria_i, bacteria_j)
            args_list = [
                (indexBacteria, allBlosumScores_local[j], allBlosumScores_local, d, w, allFitnessScores_local[j], allFitnessScores_local[indexBacteria])
                for j in range(num_bacterias) if j != indexBacteria # Excluir interacción consigo mismo
            ]

            if not args_list:
                 # print(f"Advertencia: No hay pares para calcular interacción para bacteria {indexBacteria}.")
                 total = 0.0
            else:
                # Cálculo serial (eliminado el Pool)
                total = sum(self.compute_diff(arg) for arg in args_list)


            # Asegurarse de que la suma total sea finita
            if not np.isfinite(total):
                 # print(f"Advertencia: Suma de interacción no finita para bacteria {indexBacteria}. Usando 0.0.")
                 total = 0.0

        except IndexError:
             print(f"Error de índice (al obtener scores o listas) en compute_cell_interaction para bacteria {indexBacteria}. Usando total 0.")
             total = 0.0
        except Exception as e:
            print(f"Error inesperado en compute_cell_interaction para bacteria {indexBacteria}: {e}")
            total = 0.0 # Manejo general de errores

        # Actualizar la tabla compartida
        try:
            if indexBacteria < self.numBacterias: # Asegurarse de que el índice es válido
                if atracTrue:
                    self.tablaAtract[indexBacteria] = total
                else:
                    self.tablaRepel[indexBacteria] = total
            else:
                 print(f"Advertencia: Índice {indexBacteria} fuera de rango al actualizar tabla Atract/Repel.")
        except Exception as e:
             print(f"Error inesperado actualizando tabla Atract/Repel para bacteria {indexBacteria}: {e}")


    # Métodos para crear las tablas de atracción y repulsión llamando a compute_cell_interaction (Serial)
    def creaTablaAtract(self, poblacion: List):
        """Crea la tabla de atracción para toda la población."""
        # poblacion es el Manager.list compartido (tupla de tuplas)
        print(f"    Calculando Atracción (dAttr={self.dAttr}, wAttr={self.wAttr}) (Serial)...")
        try:
            num_bacterias = len(poblacion) # Usar el tamaño de la lista compartida
            for indexBacteria in range(num_bacterias):
                 # compute_cell_interaction ya maneja si la bacteria o su score no son válidos
                 self.compute_cell_interaction(indexBacteria, self.dAttr, self.wAttr, True)
            # print("    Cálculos de Atracción completados.") # Debug print
        except Exception as e:
             print(f"Error durante la creación de la tabla de Atracción: {e}")
             # Los valores en tablaAtract pueden ser incorrectos/incompletos

    def creaTablaRepel(self, poblacion: List):
        """Crea la tabla de repulsión para toda la población."""
        # poblacion es el Manager.list compartido (tupla de tuplas)
        print(f"    Calculando Repulsión (hRep={self.hRep}, wRepel={self.wRepel}) (Serial)...")
        try:
            num_bacterias = len(poblacion) # Usar el tamaño de la lista compartida
            for indexBacteria in range(num_bacterias):
                 # compute_cell_interaction ya maneja si la bacteria o su score no son válidos
                 self.compute_cell_interaction(indexBacteria, self.hRep, self.wRepel, False)
            # print("    Cálculos de Repulsión completados.") # Debug print
        except Exception as e:
             print(f"Error durante la creación de la tabla de Repulsión: {e}")
             # Los valores en tablaRepel pueden ser incorrectos/incompletos

    # Método combinado para crear tablas de atracción y repulsión (llama a los seriales)
    def creaTablasAtractRepel(self, poblacion: List):
         """Calcula y llena las tablas de atracción y repulsión."""
         try:
             self.creaTablaAtract(poblacion)
             self.creaTablaRepel(poblacion)
             # print("    Creación de tablas de Atracción/Repulsión completada.") # Debug print
         except Exception as e:
             print(f"Error en creaTablasAtractRepel: {e}")


    # Método para crear la tabla de interacción total (Atracción + Repulsión)
    def creaTablaInteraction(self, poblacion: List): # poblacion no se usa aquí, pero se mantiene por consistencia si se llama después de otras con poblacion
        """Calcula la tabla de interacción total (atracción + repulsión) para cada bacteria."""
        try:
            len_atract = len(self.tablaAtract)
            len_repel = len(self.tablaRepel)
            target_len = self.numBacterias # Usar el tamaño esperado de la población

            # Asegurarse de que las listas compartidas tengan el tamaño esperado antes de iterar
            if len_atract != target_len or len_repel != target_len:
                 print(f"Advertencia: Tamaños de tablas de atracción/repulsión ({len_atract}, {len_repel}) no coinciden con numBacterias ({target_len}) en creaTablaInteraction.")
                 # Rellenar interaction_results con 0s si hay un problema de tamaño
                 interaction_results = [0.0] * target_len
            else:
                 interaction_results = [0.0] * target_len
                 # Sumar los términos de atracción y repulsión para cada bacteria
                 for i in range(target_len):
                     # Acceder a las listas compartidas de forma segura (ya verificamos tamaño arriba)
                     atract = self.tablaAtract[i]
                     repel = self.tablaRepel[i]

                     sum_interact = atract + repel
                     if not np.isfinite(sum_interact):
                          # print(f"Advertencia: Suma de interacción no finita para bacteria {i}. Usando 0.0.")
                          sum_interact = 0.0
                     interaction_results[i] = sum_interact

            # Actualizar la lista compartida de interacción total
            self.tablaInteraction[:] = interaction_results
            # print("    Creación de tabla de Interacción completada.") # Debug print
        except Exception as e:
            print(f"Error en creaTablaInteraction: {e}")
            # Reiniciar tablaInteraction o manejar de otra forma el error
            self.tablaInteraction[:] = [0.0] * self.numBacterias


    # Método para crear la tabla de fitness total (BLOSUM + Interacción)
    def creaTablaFitness(self, poblacion: List): # poblacion no se usa aquí, se mantiene por consistencia
        """Calcula el fitness total (score BLOSUM + interacción) para cada bacteria."""
        try:
            len_blosum = len(self.blosumScore)
            len_interact = len(self.tablaInteraction)
            target_len = self.numBacterias # Usar el tamaño esperado de la población

             # Asegurarse de que las listas compartidas tengan el tamaño esperado antes de iterar
            if len_blosum != target_len or len_interact != target_len:
                 print(f"Advertencia: Tamaños de tablas blosum/interacción ({len_blosum}, {len_interact}) no coinciden con numBacterias ({target_len}) en creaTablaFitness.")
                 # Rellenar fitness_results con -inf si hay un problema de tamaño
                 fitness_results = [-float('inf')] * target_len
            else:
                 fitness_results = [0.0] * target_len
                 # Sumar score BLOSUM y score de interacción
                 for i in range(target_len):
                     # Acceder a las listas compartidas de forma segura (ya verificamos tamaño arriba)
                     valorBlsm = self.blosumScore[i]
                     valorInteract = self.tablaInteraction[i]

                     # Asegurarse de que los valores sean finitos antes de sumar
                     if not np.isfinite(valorBlsm): valorBlsm = -float('inf') # Penalizar con -inf si no es finito
                     if not np.isfinite(valorInteract): valorInteract = 0.0 # Interacción no finita puede ser 0 o un valor penalizador

                     fitness_results[i] = valorBlsm + valorInteract # Maximizar el fitness

            # Actualizar la lista compartida de fitness total
            self.tablaFitness[:] = fitness_results
            # print("    Creación de tabla de Fitness completada.") # Debug print
        except Exception as e:
             print(f"Error en creaTablaFitness: {e}")
             # Reiniciar tablaFitness a un valor bajo para indicar fallo
             self.tablaFitness[:] = [-float('inf')] * self.numBacterias # Usar -inf si se maximiza

    # Método para obtener el valor del NFE global
    def getNFE(self) -> int:
        """Retorna el número de evaluaciones de función (NFE) acumulado."""
        try:
            # Acceder a Manager.Value directamente es seguro para lectura
            return self.globalNFE.value
        except Exception as e:
             print(f"Error accediendo a NFE global: {e}")
             return -1 # Devolver -1 o algún valor indicativo de error

    # Método para encontrar la mejor bacteria (basado en fitness)
    def obtieneBest(self) -> Tuple[int, float]:
        """Encuentra el índice y el fitness de la mejor bacteria en la población actual."""
        bestIdx = -1
        best_fitness_val = -float('inf') # Inicializar con valor mínimo si se maximiza

        try:
            fitness_list_local = list(self.tablaFitness) # Obtener una copia local para iterar de forma segura
            num_bacterias_actual = len(fitness_list_local)

            if num_bacterias_actual == 0:
                 # print("Advertencia: tablaFitness vacía en obtieneBest.")
                 return -1, -float('inf')

            # Encontrar el primer índice con fitness finito como punto de partida
            valid_indices = [i for i, f in enumerate(fitness_list_local) if np.isfinite(f)]
            if not valid_indices:
                 # print("Advertencia: No se encontró ningún fitness finito válido en obtieneBest.")
                 # Si no hay fitnesses finitos, devolver el primero (que será -inf) o un indicador.
                 return 0, fitness_list_local[0] if fitness_list_local else -float('inf')


            bestIdx = valid_indices[0]
            best_fitness_val = fitness_list_local[bestIdx]

            # Iterar sobre el resto para encontrar el máximo
            for i in valid_indices:
                 current_val = fitness_list_local[i]
                 if current_val > best_fitness_val: # Maximizar
                     best_fitness_val = current_val
                     bestIdx = i

            # Asegurarse de que bestIdx esté dentro del rango esperado de la población
            if bestIdx >= self.numBacterias:
                 print(f"Advertencia: Índice de mejor bacteria ({bestIdx}) fuera de rango ({self.numBacterias}). Ajustando.")
                 # Intentar encontrar el mejor entre los primeros numBacterias si es posible
                 if self.numBacterias > 0:
                      valid_indices_in_range = [i for i in valid_indices if i < self.numBacterias]
                      if valid_indices_in_range:
                          bestIdx = valid_indices_in_range[0]
                          best_fitness_val = fitness_list_local[bestIdx]
                          for i in valid_indices_in_range:
                               current_val = fitness_list_local[i]
                               if current_val > best_fitness_val:
                                   best_fitness_val = current_val
                                   bestIdx = i
                      else:
                           bestIdx = -1 # No se encontró mejor en el rango
                           best_fitness_val = -float('inf')
                 else:
                      bestIdx = -1
                      best_fitness_val = -float('inf')
                      print("Advertencia: numBacterias es 0.")


            # Opcional: Imprimir información sobre la mejor bacteria encontrada
            # print(f"--- Mejor Bacteria Actual Encontrada: Índice={bestIdx}, Fitness={best_fitness_val:.4f} ---")

        except Exception as e:
             print(f"Error en obtieneBest: {e}")
             import traceback
             traceback.print_exc()
             return -1, -float('inf') # Devolver indicador de error

        return bestIdx, best_fitness_val

    # Método para reemplazar la peor bacteria por la mejor (en reproducción)
    def replaceWorst(self, poblacion: List, bestIdx: int):
        """Reemplaza la bacteria con peor fitness por una copia de la mejor."""
        # poblacion es el Manager.list compartido (tupla de tuplas)
        worstIdx = -1
        worst_fitness_val = float('inf') # Inicializar con valor máximo si se maximiza

        try:
            fitness_list_local = list(self.tablaFitness) # Copia local
            num_bacterias_actual = len(fitness_list_local)

            if num_bacterias_actual == 0 or bestIdx < 0 or bestIdx >= len(poblacion):
                 print("Advertencia: No se puede ejecutar replaceWorst (datos inválidos o bestIdx fuera de rango).")
                 return

            # Encontrar el primer índice con fitness finito como punto de partida para el peor
            valid_indices = [i for i, f in enumerate(fitness_list_local) if np.isfinite(f)]
            if not valid_indices:
                 print("Advertencia: No se encontró ningún fitness finito válido para determinar el peor.")
                 return

            worstIdx = valid_indices[0]
            worst_fitness_val = fitness_list_local[worstIdx]

            # Iterar para encontrar el mínimo
            for i in valid_indices:
                 current_val = fitness_list_local[i]
                 if current_val < worst_fitness_val: # Minimizar (buscar el peor)
                     worst_fitness_val = current_val
                     worstIdx = i

            # Realizar el reemplazo si se encontró un peor índice válido diferente al mejor
            # Asegurarse de que worstIdx y bestIdx estén dentro del rango de poblacion_shared
            if worstIdx != -1 and worstIdx != bestIdx and 0 <= worstIdx < len(poblacion) and 0 <= bestIdx < len(poblacion):
                # Realizar una copia profunda de la mejor bacteria (que es una tupla de tuplas)
                if poblacion[bestIdx] is not None: # Asegurarse de que la bacteria mejor no sea None
                     poblacion[worstIdx] = copy.deepcopy(poblacion[bestIdx])

                     # Copiar también los scores y valores de las tablas correspondientes
                     # Accediendo a las listas compartidas (asegurarse de que los índices sean válidos)
                     if worstIdx < len(self.blosumScore): self.blosumScore[worstIdx] = self.blosumScore[bestIdx]
                     if worstIdx < len(self.tablaFitness): self.tablaFitness[worstIdx] = self.tablaFitness[bestIdx]
                     if worstIdx < len(self.tablaInteraction): self.tablaInteraction[worstIdx] = self.tablaInteraction[bestIdx]
                     if worstIdx < len(self.tablaAtract): self.tablaAtract[worstIdx] = self.tablaAtract[bestIdx]
                     if worstIdx < len(self.tablaRepel): self.tablaRepel[worstIdx] = self.tablaRepel[bestIdx]

                     # print(f"    Reemplazando peor bacteria (Índice {worstIdx}, Fitness {worst_fitness_val:.4f}) con copia de la mejor (Índice {bestIdx}).")
                else:
                     print(f"Advertencia: La mejor bacteria (Índice {bestIdx}) es None. No se puede reemplazar la peor.")
            elif worstIdx == bestIdx:
                 # print(f"    La mejor bacteria (Índice {bestIdx}) es también la peor o es la única válida. No se realiza reemplazo.")
                 pass # No hacer nada si la mejor es la peor
            else:
                 print(f"Advertencia: No se pudo encontrar una bacteria peor válida para reemplazar (worstIdx={worstIdx}, bestIdx={bestIdx}).")

        except IndexError:
             print(f"Error de índice durante replaceWorst (accediendo a poblacion o tablas).")
             import traceback
             traceback.print_exc()
        except Exception as e:
             print(f"Error inesperado en replaceWorst: {e}")
             import traceback
             traceback.print_exc() # Imprimir el traceback para depuración


    # Método para obtener el mejor alineamiento encontrado
    def get_best_alignment(self, bestIdx: int, poblacion: List) -> Optional[List[str]]:
        """Recupera el alineamiento de la mejor bacteria y lo formatea como lista de strings."""
        # poblacion es el Manager.list compartido (tupla de tuplas)

        # Verificar si el índice es válido y el alineamiento no es None
        if 0 <= bestIdx < len(poblacion):
            alignment_tuple: Optional[Tuple[Tuple[str, ...], ...]] = poblacion[bestIdx]
            if alignment_tuple is None:
                 print(f"Advertencia: El alineamiento para el índice {bestIdx} es None.")
                 return None # Devolver None si el alineamiento es None
            try:
                # El alineamiento se almacena como tupla de tuplas de caracteres
                # Convertir a lista de strings para salida/visualización
                alignment_strings = ["".join(seq) for seq in alignment_tuple]
                return alignment_strings
            except Exception as e:
                print(f"Error recuperando/formateando el mejor alineamiento para índice {bestIdx}: {e}")
                import traceback
                traceback.print_exc()
                return None # Devolver None si hay error
        else:
            print(f"Error: Índice del mejor ({bestIdx}) fuera de rango ({len(poblacion)}) para la población.")
            return None # Devolver None si el índice no es válido