





with open("new_1.txt",'r') as file:
    lines = [line.strip() for line in file]
print("The list is:")
print(lines)
print("The data after splitting is:")
for line in lines:
    print(line.split())
    
    
    
    
with open("new_1.txt",'r') as file:
    lines = [line.strip() for line in file]
print("The list is:")
print(lines)
print("The data after splitting is:")
for line in lines:
    print(''.join(line))
    
    

