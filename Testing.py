import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv("users_behavior.csv")
#Preparación de los datos y división en segmetos
df_train, df_temp = train_test_split(
    df,
    test_size=0.40,
    random_state=12345)
df_valid, df_test = train_test_split(
    df_temp,
    test_size=0.5,
    random_state=12345)
features_train = df_train.drop('is_ultra', axis=1)
target_train = df_train['is_ultra']

features_valid = df_valid.drop('is_ultra', axis=1)
target_valid = df_valid['is_ultra']

features_test = df_test.drop('is_ultra', axis=1)
target_test = df_test['is_ultra']
# Modelo de arbol
best_score= 0
best_depth=0
for i in range(1,4):
    model1= DecisionTreeClassifier(max_depth= i, random_state= 12345)
    model1.fit(features_train, target_train)
    a=model1.score(features_valid, target_valid)
    if a > best_score:
        best_score = a
        best_depth = i
print(f"El mejor depth de el 'árbol es: {best_depth}")
print(f"Con un score de: {best_score}")
print("\n")
#Modelo de regresión logistica
model = LogisticRegression(random_state=54321, solver='liblinear') 
model.fit(features_train, target_train)
p= model.score(features_valid, target_valid)
print(f"El score del modelo de regresión logistica es: {p}")
print("\n")
#Modelo de bosque
best_est=0
best_score=0
for est in range(1, 41):
    model = RandomForestClassifier(
        random_state=54321,
        n_estimators=est
    )
    
    model.fit(features_train, target_train)
    score = model.score(features_valid, target_valid)
    
    if score > best_score:
        best_score = score
        best_est = est
print(f"El mejor numero de estimadores de el bosque es: {best_est}")
print(f"Con un score de: {best_score}")
print("\n")
#Modelo Baseline
g = target_valid.to_frame(name="target")
mode_value = target_train.mode()[0]
g["Mode"] = mode_value
predictions = g["Mode"]
acc = accuracy_score(target_valid, predictions)
print("Accuracy del modelo baseline (moda):", acc)
print("\n")
#Evaluación final vs test
#Arbol
model1= DecisionTreeClassifier(max_depth= 3, random_state= 12345)
model1.fit(features_train, target_train)
a=model1.score(features_test, target_test)
print(f"La exactitud del mejor arbol es: {a}")
print("\n")
#Bosque
model = RandomForestClassifier(random_state=54321,n_estimators= 40)    
model.fit(features_train, target_train)
b= model.score(features_test, target_test)
print(f"La exactitud del mejor bosque es de: {b}")
print("\n")
#Logística
model = LogisticRegression(random_state=54321, solver='liblinear') 
model.fit(features_train, target_train)
c= model.score(features_test, target_test)
print(f"El exactitud del modelo de regresión es: {c}")
print("\n")
#Baseline
d= accuracy_score(target_test, predictions)
print(f"La exactitud del modelo aleatorio es: {d}")
print("\n")
print("Se probaron distintos modelos de clasificación para recomendar el plan correcto a los usuarios. El modelo con mejor desempeño fue RandomForestClassifier, alcanzando una exactitud de 0.79 en el conjunto de prueba, superando el umbral requerido de 0.75. El árbol de decisión también mostró buenos resultados, aunque ligeramente inferiores. La regresión logística obtuvo menor exactitud debido a su naturaleza lineal. El modelo baseline basado en la moda presentó un rendimiento inferior, lo que confirma que los modelos entrenados capturan patrones reales en los datos.")