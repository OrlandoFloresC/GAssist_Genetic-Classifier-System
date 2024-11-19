import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

# Cargar el dataset
datos = pd.read_csv('C:/path/to/your/dataset/data.csv') #pc

# Separar las características de la clase
X = datos.iloc[:, :-1]  # Características (todas las columnas excepto la última)
y = datos.iloc[:, -1]   # Clase (última columna)

# Discretizar todas las características usando OneHot Encoding
n_bins = 3  # Número de intervalos a crear por cada característica
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='onehot', strategy='uniform')
X_discretized_sparse = discretizer.fit_transform(X)

# Convertir la matriz dispersa a una matriz densa
X_discretized = X_discretized_sparse.toarray()

# Crear nombres de columnas para el nuevo DataFrame discretizado
n_features = X_discretized.shape[1]
column_names = ['X' + str(i) for i in range(n_features)]

# Combinar las características discretizadas con la clase
datos_discretized = pd.DataFrame(X_discretized, columns=column_names)
datos_discretized['Class'] = y.values

# Dividir el dataset en entrenamiento y prueba (opcional)
trainData, testData = train_test_split(datos_discretized, test_size=0.2, random_state=42)

######################################################################################
# Solo para sacar una cantidad deseada de observaciones (filas)
# Definir el número deseado de observaciones para cada conjunto
#n_train_observations = 40
#n_test_observations = 10

# Filtrar el número de observaciones para cada dataset
#trainData = trainData.iloc[:n_train_observations]
#testData = testData.iloc[:n_test_observations]

################################################################################################

# Función para guardar los DataFrames alineados
def save_dataframe_aligned(df, filename):
    with open(filename, 'w') as f:
        # Escribir los nombres de las columnas separados por tabulaciones
        f.write('\t'.join(df.columns) + '\n')
        
        # Escribir cada fila de datos separados por tabulaciones
        for _, row in df.iterrows():
            formatted_row = '\t'.join([str(int(val)) if isinstance(val, (float, int)) and val.is_integer() else str(val) for val in row])
            f.write(formatted_row + '\n')

# Guardar los archivos alineados
save_dataframe_aligned(trainData, 'train_data_aligned.txt')
save_dataframe_aligned(testData, 'test_data_aligned.txt')
