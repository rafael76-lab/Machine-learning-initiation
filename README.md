# Machine-learning-initiation
This is the first academic project i realized with AI
La tarea era crear un modelo de machine learning supervisado para predecir el plan óptimo (Smart o Ultra) para clientes de la empresa Megaline, a partir de su comportamiento de uso.

Trabajé con datos que previamente procesé y realicé la segmentación en conjuntos de entrenamiento, validación y prueba (60/20/20) para asegurar una correcta evaluación del desempeño del modelo. Implementé y comparé distintos algoritmos de clasificación, incluyendo árboles de decisión, regresión logística y Random Forest, optimizando sus hiperparámetros mediante prueba, validación y edición.

El modelo final seleccionado fue un RandomForestClassifier, que alcanzó una exactitud de 0.79 en el conjunto de prueba, superando el umbral requerido de 0.75. Se validó el desempeño contra un modelo baseline basado en la moda, confirmando que el modelo captura patrones significativos en los datos.
