from creadto.services.classification import classify_from_directory, classify_from_file

print(classify_from_file("./sample/m-daniel-half.jpeg"))
print(classify_from_file("./sample/f-jenny-full.jpeg"))
print(classify_from_file("./sample/f-moon-half.jpg"))

print(classify_from_directory("./sample"))