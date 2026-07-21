import os,shutil

"""
Copies over
"""

webDependencies = ["computer.py","computer_supervised.py","computerRL.py","game.py","genetic_model.py","main.py","minimax_model.py","othello_engine.py"]


files = os.listdir(os.getcwd()+"/web")
missing = [file for file in webDependencies if file not in files]
for file in missing:
    shutil.copy(file,os.getcwd()+"/web/"+file)
    print(f"Added \"{file}\"!")

