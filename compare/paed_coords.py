from useful_functs import functions as F


root = r'/home/oli/data/results/oli/run_folder/eval_100_3'
pred_coords_1 = F.load_obj_pickle(root, 'final_coords_no_struc')
pred_coords_2 = F.load_obj_pickle(root, 'final_coords_no_struc_2')


print(pred_coords_1)

print('---------------------------------')

print(pred_coords_2)
