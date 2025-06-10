age = int(input("Enter your age: "))
country = str(input("Enter your country: "))
country2 = str(input("Enter your country again: "))
if country != country2:
    print("escribiste mal el país")


if age >= 18 and country == "Chile":
    print("You are eligible to vote in Chile.")
else:
    print("You are not eligible to vote in Chile.")    
    