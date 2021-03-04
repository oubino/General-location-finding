import os
import csv


root = r'C:\Users\olive\OneDrive\Documents\Photomed Essay'

files = list(sorted(os.listdir(root)))
print(files)

with open(r'C:\Users\olive\OneDrive\Documents\CNN\db.csv', 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      for i in files:
          writer.writerow([i])
        