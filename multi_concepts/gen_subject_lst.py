import os

data_root = "./data/PuzzleIOI/puzzle"

with open("./clusters/subject_all.txt", "w") as f:
    for subject in os.listdir(data_root):
        for motion in os.listdir(os.path.join(data_root, subject)):
            file_num = len(os.listdir(os.path.join(data_root, subject, motion, 'images')))
            if file_num != 101:
               print(subject, motion, file_num)
            f.write(f"{subject}/{motion} {subject}_{motion}\n")