from matplotlib import pyplot as plt

def show_img(IMG, fig_size=(18,10)):
    plt.figure(figsize=fig_size); plt.axis('off')
    plt.imshow(IMG ,cmap='gray')
    plt.show()
    
def show_img_s(IMG, titles=None, fig_size = (18,10)):
    img_qty = len(IMG)
    plt.figure(figsize=fig_size); 
    for i in range(img_qty):
        plt.subplot(1,img_qty, i+1)
        if titles != None:
            plt.title(titles[i])
        plt.imshow(IMG[i] ,cmap='gray'); plt.axis('off')
    plt.tight_layout()
    plt.show()
