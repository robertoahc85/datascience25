coordenas=[(122,467),(56,78),(12,45),(67,89),(34,56)]
for punto in coordenas:
    print(f"Coordenada: {punto[0]}, {punto[1]})")
coordenas.add((100,200))  # Esto no funcionará porque las tuplas son inmutables

