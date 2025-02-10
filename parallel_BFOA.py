from copy import copy
from multiprocessing import Manager, Pool
import time
from bacteria import bacteria
import numpy
import copy

from fastaReader import fastaReader

if __name__ == "__main__":
    numeroDeBacterias = 4
    numRandomBacteria = 1
    iteraciones = 3
    tumbo = 2                                             #numero de gaps a insertar 
    nado = 3
    secuencias = list()
    
    secuencias = fastaReader().seqs
    names = fastaReader().names
    
        
    
  
         
    
    
    #hace todas las secuencias listas de caracteres
    for i in range(len(secuencias)):
        #elimina saltos de linea
        secuencias[i] = list(secuencias[i])
        

    

    globalNFE = 0                            #numero de evaluaciones de la funcion objetivo
    
    

    dAttr= 0.1 #0.1
    wAttr= 0.002 #0.2
    hRep=dAttr
    wRep= .001    #10
    
   

  
    
    manager = Manager()
    numSec = len(secuencias)
    print("numSec: ", numSec)
    
    poblacion = manager.list(range(numeroDeBacterias))
    names = manager.list(names)
    NFE = manager.list(range(numeroDeBacterias))
    
    
    # print(secuencias)



    def poblacionInicial():    #lineal
        #crece la poblacion al numero de bacterias
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
    veryBest = [None, None, None] #indice, fitness, secuencias
    
    #registra el tiempo de inicio
    start_time = time.time()
    
    print("poblacion inicial ...")
    poblacionInicial() 
    
    for it in range(iteraciones):
        print("poblacion inicial creada - Tumbo ...")
        operadorBacterial.tumbo(numSec, poblacion, tumbo)
        print("Tumbo Realizado - Cuadrando ...")
        operadorBacterial.cuadra(numSec, poblacion)
        print("poblacion inicial cuadrada - Creando granLista de Pares...")
        operadorBacterial.creaGranListaPares(poblacion)
        print("granList: creada - Evaluando Blosum Parallel")
        operadorBacterial.evaluaBlosum()  #paralelo
        print("blosum evaluado - creando Tablas Atract Parallel...")

        operadorBacterial.creaTablasAtractRepel(poblacion, dAttr, wAttr,hRep, wRep)


        operadorBacterial.creaTablaInteraction()
        print("tabla Interaction creada - creando tabla Fitness")
        operadorBacterial.creaTablaFitness()
        print("tabla Fitness creada ")
        globalNFE += operadorBacterial.getNFE()
        bestIdx, bestFitness = operadorBacterial.obtieneBest(globalNFE)
        if (veryBest[0] == None) or (bestFitness > veryBest[1]): #Remplaza el mejor 
            veryBest[0] = bestIdx
            veryBest[1] = bestFitness
            veryBest[2] = copy.deepcopy(poblacion[bestIdx])
        operadorBacterial.replaceWorst(poblacion, veryBest[0])
        operadorBacterial.resetListas(numeroDeBacterias)

    print("Very Best: ", veryBest)
    #imprime el tiempo de ejecucion
    print("--- %s seconds ---" % (time.time() - start_time))



    