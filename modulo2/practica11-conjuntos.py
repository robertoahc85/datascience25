frutas= {'apple', 'banana', 'cherry', 'date', 'elderberry'}
print(frutas)

A={5, 10, 15, 20, 25}
B={10, 20, 30, 40, 50}
print("union", A|B)  # Unión de conjuntos
A.add(35)  # Agrega un elemento al conjunto A
print("interseccion", A&B)  # Intersección de conjuntos
B.remove(30)  # Elimina un elemento del conjunto B
print("union", A|B)  # Unión de conjuntos
print("diferencia", A-B)  # Diferencia de conjuntos
print("diferencia simetrica", A^B)  # Diferencia simétrica de conjuntos
print("A es un subconjunto de B:", A.issubset(B))  # Verifica si A es un subconjunto de B
print("B es un superconjunto de A:", B.issuperset(A))  # Verifica si B es un superconjunto de A
print("A es disjunto de B:", A.isdisjoint(B))  # Verifica si A y B son disjuntos