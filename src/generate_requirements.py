import subprocess

def create_req():
    ignore = ["'","", " ", "\n", "Package", "Version", "---------", "=="]
    valid = []
    modules = []
    sp = subprocess.Popen(['pip','list'], stdout=subprocess.PIPE)
    output, error = sp.communicate()
    output = str(output)
    list = output.split()
    for item in list[3:]:
        chars = item.split("\\n")
        for char in chars:
            if char not in ignore:
                valid.append(char)
    for i,V in enumerate(valid):
        if i%2 == 0:
            modules.append(V+"=="+valid[i+1]+"\n")
    with open("requirements.txt","w") as req:
        req.writelines(modules)
    req.close()

create_req()
print("requirements.txt file successfully created in local directory.")