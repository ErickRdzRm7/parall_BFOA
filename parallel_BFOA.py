from copy import copy
from multiprocessing import Manager, Pool
import time
from bacteria import bacteria
import numpy
import copy
import pandas as pd
import matplotlib.pyplot as plt

from fastaReader import fastaReader

if __name__ == "__main__":
    numeroDeBacterias = 3
    numRandomBacteria =1
    iteraciones = 3
    tumbo = 2  # numero de gaps a insertar 
    nado = 4
    secuencias = list()
    resultados = []
    secuencias = fastaReader().seqs
    names = fastaReader().names
    
    # hace todas las secuencias listas de caracteres
    for i in range(len(secuencias)):
        secuencias[i] = list(secuencias[i])

    globalNFE = 0  # numero de evaluaciones de la funcion objetivo

    dAttr = 0.1  # 0.1
    wAttr = 0.002  # 0.2
    hRep = dAttr
    wRep = .001  # 10

    manager = Manager()
    numSec = len(secuencias)
    print("numSec: ", numSec)

    poblacion = manager.list(range(numeroDeBacterias))
    names = manager.list(names)
    NFE = manager.list(range(numeroDeBacterias))

    def poblacionInicial():  # lineal
        for i in range(numeroDeBacterias):
            bacterium = []
            for j in range(numSec):
                bacterium.append(secuencias[j])
            poblacion[i] = list(bacterium)

    def printPoblacion():
        for i in range(numeroDeBacterias):
            print(poblacion[i])

    #---------------------------------------------------------------------------------------------------------
    operadorBacterial = bacteria(numeroDeBacterias)    
    veryBest = [None, None, None]  # indice, fitness, secuencias

    # registra el tiempo de inicio
    start_time = time.time()
    
    print("poblacion inicial ...")
    poblacionInicial() 
    
    # Inicialización de listas para almacenar el progreso
    mejores_fitness = []
    mejores_indices = []

    for it in range(iteraciones):
        iter_start_time = time.time()

        print("poblacion inicial creada - Tumbo ...")
        operadorBacterial.tumbo(numSec, poblacion, tumbo)
        print("Tumbo Realizado - Cuadrando ...")
        operadorBacterial.cuadra(numSec, poblacion)
        print("poblacion inicial cuadrada - Creando granLista de Pares...")
        operadorBacterial.creaGranListaPares(poblacion)
        print("granList: creada - Evaluando Blosum Parallel")
        operadorBacterial.evaluaBlosum()  # paralelo
        print("blosum evaluado - creando Tablas Atract Parallel...")

        operadorBacterial.creaTablasAtractRepel(poblacion, dAttr, wAttr, hRep, wRep)
        operadorBacterial.creaTablaInteraction()
        print("tabla Interaction creada - creando tabla Fitness")
        operadorBacterial.creaTablaFitness()
        print("tabla Fitness creada")

        globalNFE += operadorBacterial.getNFE()

        # Obtener el mejor fitness y su índice
        bestIdx, bestFitness = operadorBacterial.obtieneBest(globalNFE)
        mejores_fitness.append(bestFitness)
        mejores_indices.append(bestIdx)

    # Crear un DataFrame con los resultados
    df_resultados = pd.DataFrame({
        'Iteración': list(range(1, iteraciones + 1)),
        'Índice Mejor Bacteria': mejores_indices,
        'Mejor Fitness': mejores_fitness
    })

    print(df_resultados)

    # Graficar el progreso del mejor fitness
    plt.figure(figsize=(10, 6))
    plt.plot(df_resultados['Iteración'], df_resultados['Mejor Fitness'], marker='o', linestyle='-', color='b')
    plt.title("Progreso del Mejor Fitness en Cada Iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Mejor Fitness")
    plt.grid(True)
    plt.show()

    # Guardar los resultados en un archivo CSV
    df_resultados.to_csv("resultados_bacteria.csv", index=False)
