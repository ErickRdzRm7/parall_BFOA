import numpy

class fastaReader():

    # Modificado para aceptar el path del archivo
    def __init__(self, path: str):
        self.path = path # Usar el path pasado como argumento

        self.seqs = list()
        self.names = list()
        self.read()

    def read(self):
        """Lee el archivo FASTA y carga secuencias y nombres."""
        try:
            f = open(self.path, "r")
            lines = f.readlines()
            f.close()
        except FileNotFoundError:
            print(f"Error: Archivo no encontrado en el path: {self.path}")
            return # Salir si el archivo no existe
        except Exception as e:
            print(f"Error leyendo el archivo FASTA {self.path}: {e}")
            return # Manejar otros errores de lectura

        seq = ""
        for line in lines:
            line = line.strip() # Eliminar espacios en blanco al inicio y final
            if not line: continue # Saltar líneas vacías

            if line.startswith(">"): # Identificador de secuencia
                # Guardar la secuencia previa si existe
                if seq != "":
                    self.seqs.append(seq)
                # Guardar el nombre (sin el '>')
                self.names.append(line[1:])
                seq = "" # Reiniciar la secuencia
            else:
                # Concatenar líneas de secuencia
                seq += line

        # No olvides añadir la última secuencia después del bucle
        if seq != "":
            self.seqs.append(seq)

    # Método opcional para mostrar secuencias y nombres (para depuración)
    def show_sequences(self):
        print("Nombres de Secuencias:")
        for name in self.names:
            print(f"- {name}")
        print("\nSecuencias:")
        for i, seq in enumerate(self.seqs):
             print(f"Secuencia {i+1}: {seq[:60]}...") # Imprimir solo los primeros 60 chars

    pass # La palabra 'pass' al final de la clase no es necesaria si ya hay métodos