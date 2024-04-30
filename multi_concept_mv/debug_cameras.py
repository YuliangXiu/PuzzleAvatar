from jutils import mesh_utils, plot_utils, image_utils

import os
import os.path as osp

import pickle 

a_name = 'camera_list_mv_unet.pkl'
b_name = 'camera_list_mv_pipeline.pkl'

device = 'cuda:0'
with open(a_name, 'rb') as f:
    a = pickle.load(f)
with open(b_name, 'rb') as f:
    b = pickle.load(f)
    

# cam_a = plot_utils.vis_cam(wTc=a[0].cuda()[::20], 
#                            size=.2, 
#                            color='red')
# cam_a += plot_utils.vis_cam(wTc=a[1].cuda()[::20], 
#                            size=.2, 
#                            color='blue')
# cam_a += plot_utils.vis_cam(wTc=a[2].cuda()[::20], 
#                            size=.2, 
#                            color='yellow')
# cam_a += plot_utils.vis_cam(wTc=a[3].cuda()[::20], 
#                            size=.2, 
#                            color='white')

a = a.float().reshape(-1, 4, 4, 4)[0]
cam_a = plot_utils.vis_cam(wTc=a[0].cuda(), 
                           size=.2, 
                           color='red')
cam_a += plot_utils.vis_cam(wTc=a[1].cuda(), 
                           size=.2, 
                           color='blue')
cam_a += plot_utils.vis_cam(wTc=a[2].cuda(), 
                           size=.2, 
                           color='yellow')
cam_a += plot_utils.vis_cam(wTc=a[3].cuda(), 
                           size=.2, 
                           color='white')
coord = plot_utils.create_coord(device)
scene = mesh_utils.join_scene(cam_a + [coord])

image_list = mesh_utils.render_geom_rot_v2(scene)
image_utils.save_gif(image_list, './camera_mc_unet')


# ########
b = b.float()
cam_b = plot_utils.vis_cam(wTc=b[1:2].cuda(), color='red', size=.2)
cam_b += plot_utils.vis_cam(wTc=b[2:3].cuda(), color='blue', size=.2)
cam_b += plot_utils.vis_cam(wTc=b[3:4].cuda(), color='yellow', size=.2)
cam_b += plot_utils.vis_cam(wTc=b[0:1].cuda(), color='white', size=.2)

# cam_a = plot_utils.vis_cam(cTw=a.cuda(),)
# cam_b = plot_utils.vis_cam(cTw=b.cuda(), color='red')

coord = plot_utils.create_coord(device)
scene = mesh_utils.join_scene(cam_b + [coord])

image_list = mesh_utils.render_geom_rot_v2(scene)
image_utils.save_gif(image_list, './camera_mv_pipeline')