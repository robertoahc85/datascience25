# number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# even_number = []
# for contador in number:
#     if contador % 2 == 0:
#         even_number.append(contador)
# print("Los números pares son:", even_number) 

# letters = ['a', 'b', 'c', 'd', 'e']
# reversed_letters = []
# for i in range(len(letters)-1, -1, -1):
#     reversed_letters.append(letters[i])
# print(reversed_letters)   


# range_number2= range(11,1,-1)
# for i in range_number2:
#     print(i)
# print(type(range_number2))    

# number = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# total=0
# for i in number:
#     total+= i
# print(total)  

number2= [1, 2, 7, 4, 5, 7, 7, 8, 9, 10] 
unique_numbers = []
for i in number2:
    if i not in unique_numbers:
        unique_numbers.append(i)
print("Los números únicos son:", unique_numbers)  

suma = sum(number2)  
suma_unicos = sum(unique_numbers)
print("La suma de los números únicos es:", suma_unicos) 
print("La suma de los números es:", suma)   #snakecase camelcase
