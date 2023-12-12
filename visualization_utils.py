import matplotlib.pyplot as plt
import shutil
import numpy as np
import matplotlib as mpl
from astropy.visualization import simple_norm
import os
from PIL import Image


def matrix_plot_from_folder(folder_path,save_path):
    """Takes in a folder with .npy images and returns a default grid of images
    For more flexibility, see matrix_plot_from_npy
    
    Args:
        folder_path (string): path to folder w/ .npy images created by paltas
        save_path (string): path to save final image to
    Returns:

    """
    file_list = []
    names=[]
    for i in range(0,100):
        file_list.append(folder_path+'image_%07d.npy'%(i))
        names.append(str(i))

    matrix_plot_from_npy(file_list[:40],names,(4,10),save_path,annotate=False)

###############################
# Matrix plot from numpy files
###############################
def matrix_plot_from_npy(file_list,names,dim,save_name,annotate=False):
    """
    Args: 
        file_list ([string]): paths of .npy files
        names ([string]): labels for images
        dim (int,int): Tuple of (number rows, number columns), defines shape of 
            matrix plotted
        save_name (string): Filename to save final image to
    """

    # edge case: not enough spaces given for # of images
    if dim[0]*dim[1] < len(file_list):
        print("Matrix not big enough to display all lens images." + 
            "Retry with larger matrix dimensions. Exiting matrix_plot()")
        return

    # TODO: check to see if this folder already exists
    os.mkdir('intermediate_temp')

    # prevent matplotlib from showing intermediate plots
    backend_ =  mpl.get_backend() 
    mpl.use("Agg")  # Prevent showing stuff

    file_counter = 0
    completed_rows = []
    offset = 40
    for i in range(0,dim[0]):
        row = []
        for j in range(0,dim[1]):
            if file_list[file_counter] is None:
                cropped_data = np.ones((80,80))
            else:
                cropped_data = np.load(file_list[file_counter])
                # normalize data using log and min cutoff 
            #norm = simple_norm(cropped_data,stretch='log',min_cut=1e-6)
        
            # create individual image using plt library
            if names[file_counter] is not None:
                plt.matshow(cropped_data,cmap='viridis',norm='asinh')
            else:
                plt.matshow(cropped_data,cmap='Greys_r')

            if annotate and file_list[file_counter] is not None:
                # annotate system name
                plt.annotate(names[file_counter],(2*offset - offset/8,offset/6),color='white',
                    fontsize=20,horizontalalignment='right')
                # show size of 1 arcsec
                plt.plot([offset/6,offset/6],[offset/8,
                    offset/8 + (1/0.04)],color='white')
                if file_counter == 0:
                    plt.annotate('1"',(offset/6 +2,offset/2),color='white',fontsize=20)
            
            plt.axis('off')

            # save intermediate file, then read back in as array, and save to row
            if names[file_counter] is None:
                intm_name = 'intermediate_temp/none.png'
            else:
                intm_name = ('intermediate_temp/'+ names[file_counter]
                    +'.png')
            plt.savefig(intm_name,bbox_inches='tight',pad_inches=0)
            img_data = np.asarray(Image.open(intm_name))
            plt.close()
            row.append(img_data)
            # manually iterate file index
            file_counter += 1

        # stack each row horizontally in outer loop
        # edge case: one column
        if dim[1] == 1:
            build_row = row[0]
        else:
            build_row = np.hstack((row[0],row[1]))

        if dim[1] > 2:
            for c in range(2,dim[1]):
                build_row = np.hstack((build_row,row[c]))
            completed_rows.append(build_row)

    # reset matplotlib s.t. plots are shown
    mpl.use(backend_) # Reset backend
    
    # clean up intermediate files
    shutil.rmtree('intermediate_temp')

    # stack rows to build final image
    # edge case: one row
    print(np.shape(completed_rows[0]))
    print(np.shape(completed_rows[1]))
    if dim[0] == 1:
        final_image = completed_rows[0]
    else:
        final_image = np.vstack((completed_rows[0],completed_rows[1]))

    if dim[0] > 2:
        for r in range(2,dim[0]):
            final_image = np.vstack((final_image,completed_rows[r]))

    # plot image and save
    plt.figure(figsize=(2*dim[1],2*dim[0]))
    plt.imshow(final_image)
    plt.axis('off')
    plt.savefig(save_name,bbox_inches='tight')
    plt.show()