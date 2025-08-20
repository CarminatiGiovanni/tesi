import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
matplotlib.use('TkAgg')


# Load your image
img = mpimg.imread("images/scimmia.jpeg")



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

# Show images
im1 = ax1.imshow(img, cmap='hot', origin='upper')
ax1.set_title('Image 1')
fig.colorbar(im1, ax=ax1)  # optional colorbar

def onclick(event):
    if event.inaxes == ax1:
        if event.xdata is None or event.ydata is None:
            return  # click outside axes
        # Convert to integer pixel indices
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        # Make sure indices are inside image bounds
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            pixel_value = img[y, x]  # note: rows=y, cols=x
            ax2.clear()
            ax2.bar(['R','G','B'],pixel_value, color=['red', 'green', 'blue'])
            plt.show()
            print(f"Clicked pixel at ({x}, {y})")  # update the figure

# Connect the event
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

