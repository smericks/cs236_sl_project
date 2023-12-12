import matplotlib.pyplot as plt
import shutil
import numpy as np
import matplotlib as mpl
from astropy.visualization import simple_norm
import os
from PIL import Image
import corner
from matplotlib.lines import Line2D

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


def overlay_contours(emcee_chains_list,colors_list,iofi,
    param_labels,sampler_labels,true_params=None,save_path=None,bounds=None,
    burnin=int(1e3),annotate_medians=False,dpi=200):
    """
    Args:
        emcee_chains_list (list): list of chains from emcee sampler
        colors_list (list): list of colors for each contour
        iofi (list[int]): list of indices of parameters to be plotted
        true_params ([float]): ground truth for iofi ONLY
        param_labels (list[string]): list of labels for each parameter of interest
        sampler_labels (list[string]): list of labels for each sampler/contour
        bounds (list): list of [min,max] bounds for each param's contour
    """
    import copy


    corner_kwargs = {
        'labels':np.asarray(param_labels),
        'bins':20,
        'show_titles':False,
        'plot_datapoints':False,
        'label_kwargs':dict(fontsize=25),
        'levels':[0.68,0.95],
        'color':colors_list[0],
        'fill_contours':True,
        'contourf_kwargs':{},
        'hist_kwargs':{'density':True,'color':colors_list[0],
                       'lw':3},
        'title_fmt':'.2f',
        'plot_density':False,
        'max_n_ticks':3,
        'range':np.ones(len(iofi))*0.98,
        'smooth':1.
    }

    if true_params is not None:
        corner_kwargs['truths'] = true_params
        corner_kwargs['truth_color'] = 'black'

    figure = plt.figure(dpi=dpi,figsize=(12,12))

    for i,emcee_chain in enumerate(emcee_chains_list):
        if i == 0:
            corner_kwargs_copy = copy.deepcopy(corner_kwargs)
            figure = param_of_interest_corner(emcee_chain,iofi,corner_kwargs_copy,
                burnin=burnin,figure=figure)
        else:
            corner_kwargs_copy = copy.deepcopy(corner_kwargs)
            corner_kwargs_copy['color'] = colors_list[i]
            corner_kwargs_copy['hist_kwargs']['color'] = colors_list[i] 
            figure = param_of_interest_corner(emcee_chain,iofi,corner_kwargs_copy,
                figure=figure,burnin=burnin)

    num_params = len(iofi)
    axes = np.array(figure.axes).reshape((num_params, num_params))
    custom_lines = []
    for color in colors_list:
        custom_lines.append(Line2D([0], [0], color=color, lw=6))

    axes[0,num_params-1].legend(custom_lines,sampler_labels,frameon=False,
                fontsize=20,loc=7)
    #axes[0,num_params-1].legend(custom_lines,sampler_labels,
    #    frameon=False,fontsize=30)
    
    for r in range(0,num_params):
        for c in range(0,r+1):
            if bounds is not None:
                axes[r,c].set_xlim(bounds[c])
                if r != c :
                    axes[r,c].set_ylim(bounds[r])

            if annotate_medians and r==c:
                legend_text_list = []
                for emcee_chain in emcee_chains_list:

                    num_params = emcee_chain.shape[-1]
                    if (len(emcee_chain.shape) == 3):
                        chain = emcee_chain[:,burnin:,:].reshape((-1,num_params))
                    else:
                        chain = emcee_chain[burnin:,:]
                        
                    
                    med = np.median(chain[:,iofi[c]])
                    low = np.quantile(chain[:,iofi[c]],q=0.1586)
                    high = np.quantile(chain[:,iofi[c]],q=0.8413)

                    legend_text_list.append('%.2f $\pm^{%.2f}_{%.2f}$'%(med,high-med,med-low))

                axes[r,c].legend(custom_lines,legend_text_list,
                    frameon=True,fontsize=12,handlelength=1,loc='lower right')
                #axes[r,c].set_title(param_labels[c])

    if save_path:
        plt.savefig(save_path)

    return figure

def param_of_interest_corner(emcee_chain,iofi,corner_kwargs,true_params=None,
                             burnin=int(1e3),bounds=None,title=None,
                             figure=None,display_metric=False):
    """
    Args: 
        emcee_chain (array[n_walkers,n_samples,n_params])
        iofi ([int]): list of indices of which params to plot
        corner_kwargs (dict): corner.corner() arguments
        true_params (list[float]): list of ground truth ONLY for iofi
    Returns:
        matplotlib figure object
    """
    num_params = emcee_chain.shape[-1]
    if (len(emcee_chain.shape) == 3):
        chain = emcee_chain[:,burnin:,:].reshape((-1,num_params))
    else:
        chain = emcee_chain[burnin:,:]
    # subtract off the ground truth for params of interest
    if true_params is not None:
        for j,i in enumerate(iofi):
            chain[:,i] -= true_params[j]

    print(chain.shape)
    if figure is None:
        figure = corner.corner(chain[:,iofi],fig=None,**corner_kwargs)
    else:
        figure = corner.corner(chain[:,iofi],fig=figure,**corner_kwargs)

    num_params = len(iofi)
    axes = np.array(figure.axes).reshape((num_params, num_params))
    #custom_lines = [Line2D([0], [0], color=COLORS['hyperparam_narrow'], lw=2),
    #    Line2D([0], [0], color='grey', lw=2)]

    #axes[0,num_params-1].legend(custom_lines,['Narrow','Truth'],frameon=False,fontsize=12)

    if title is not None:
        plt.suptitle(title,fontsize=20)


    if bounds is not None:

        for r in range(0,num_params):
            for c in range(0,r+1):
                axes[r,c].set_xlim(bounds[c])
                if r != c :
                    axes[r,c].set_ylim(bounds[r])
                else:
                    if display_metric:
                        c_idx = iofi[c]
                        med = np.median(chain[:,c_idx])
                        low = np.quantile(chain[:,c_idx],q=0.1586)
                        high = np.quantile(chain[:,c_idx],q=0.8413)
                        axes[r,c].legend(custom_lines,
                            ['%.2f $\pm^{%.2f}_{%.2f}$'%(med,high-med,med-low),'%.2f'%(true_params[c])],
                            frameon=True,fontsize=14,handlelength=1,loc='lower right')

    #plt.show()

    return figure