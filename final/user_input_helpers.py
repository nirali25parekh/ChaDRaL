# input image file name from user
def image_input_from_user():                           #returns path to selected image
    path = "NULL" 
    path = input("Please enter file name:  ")         # show an "Open" dialog box and return the path to the selected file
    if path!="NULL":
        print("Image loaded!") 
    else:
        print("Error Image not loaded!")
    return path

# input key from user
def key_input_from_user():
    key = input("Enter key: ")
    return key
