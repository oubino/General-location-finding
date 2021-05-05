from useful_functs import functions as F
import math
root = r'/home/oli/data/results/oli/run_folder/eval_100_3'
pred_coords_1 = F.load_obj_pickle(root, 'final_coords_no_struc')
pred_coords_2 = F.load_obj_pickle(root, 'final_coords_no_struc_2')


#print(pred_coords_1)

print('---------------------------------')

#print(pred_coords_2)


dev_list = {}
dev_list_x = {}
dev_list_y = {}
dev_list_z = {}

mean_dev = {}
mean_dev_x = {}
mean_dev_y = {}
mean_dev_z = {}
mean_dev_std = {}
mean_list = {}

landmarks = [1,2,3,4,5,6,7,8,9,10]
for i in pred_coords_1.keys():
    for l in landmarks:
        x_1 = pred_coords_1[i][l]['x']
        x_2 = pred_coords_2[i][l]['x']
        y_1 = pred_coords_2[i][l]['y']
        y_2 = pred_coords_2[i][l]['y']
        z_1 = pred_coords_2[i][l]['z']
        z_2 = pred_coords_2[i][l]['z']
        
        x_dev = x_1 - x_2
        y_dev = y_1 - y_2
        z_dev = z_1 - z_2
        dev = math.sqrt(abs(x_dev)**2 + abs(y_dev)**2 + abs(z_dev)**2)
        print('i: ' + str(i) + 'l: ' + str(l) + 'dev:' + str(dev))
 #       dev_list['%1.0f' % i].append(dev)
  #      dev_list_x['%1.0f' % i].append(x_dev)
   #     dev_list_y['%1.0f' % i].append(y_dev)
    #    dev_list_z['%1.0f' % i].append(z_dev)
        
        
