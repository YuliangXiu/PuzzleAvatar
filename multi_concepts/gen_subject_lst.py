import os

data_root = "./data/PuzzleIOI/puzzle"

with open("./clusters/subject_all.txt", "w") as f:
    for subject in os.listdir(data_root):
        for motion in os.listdir(os.path.join(data_root, subject)):
            file_num = len(os.listdir(os.path.join(data_root, subject, motion, 'images')))
            if file_num != 101:
               print(subject, motion, file_num)
            f.write(f"PuzzleIOI/puzzle/{subject}/{motion} {subject} {motion}\n")
            os.rename(os.path.join(data_root, subject, motion, "images"), 
                      os.path.join(data_root, subject, motion, "image"))
            # os.makedirs(f"./logs/{subject}/{motion}", exist_ok=True)