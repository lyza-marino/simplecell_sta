'''Simple cell RF implementation using a 2D Gabor Function, by Lyza Marino'''
import numpy as np
import matplotlib.pyplot as plt

#Set up +-5 degree x and y, discretize so that 1 pixel corresponds to 0.2 deg x 0.2 deg (50 x 50 image)
X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
x,y = np.meshgrid(X,Y)

#Params
σx = 1       #degrees
σy = 2       #degrees
k  = 1/0.56  #degrees
ϕ  = np.pi/2 

#Simple cell with a vertical RF given by a 2D Gabor function
RF =(1/(2*np.pi*σx*σy))*(np.e**((-(x**2/(2*σx**2)))-(y**2/(2*σy**2)))*np.cos(k*x-ϕ))
#Plot RF as a 2D image
plt.imshow(RF, cmap = 'bwr', extent=[-5,5,-5,5])
plt.colorbar(label = 'RF Amplitude')
plt.xlabel('x (deg)')
plt.title("RF as a 2D Image")
plt.ylabel('y (deg)')
plt.show()

#Plot RF as a cross-section at y=0 (along x axis)
plt.figure()
plt.plot(X, RF[RF.shape[0]//2, :])
plt.title('Cross Section at y = 0')
plt.xlabel('x (deg)')
plt.ylabel('RF')
plt.show()

#Spike rate as a response to vertical grating images
peak_rate = 50
alphas = np.arange(0, 2*np.pi, 0.01)
responses = [0 for i in range(len(alphas))]
for i in range(len(alphas)):
    I = np.sin(k * x - alphas[i])
    RFtimesImage = np.dot(RF.flatten(), I.flatten())    
    hw_nonlinearity = max(0, RFtimesImage)**2
    responses[i] = hw_nonlinearity
    
responses = peak_rate * (responses / np.max(responses))

plt.imshow(np.sin(k * x - 0), cmap='gray', extent=[-5,5,-5,5])
plt.title('Vertical Grating, Phase = 0')
plt.xlabel("x (deg)")
plt.ylabel("y (deg)")
plt.show()

plt.plot(alphas, responses)
plt.ylabel("Firing Rate (spikes/second)")
plt.xlabel("Grating Phase (0 to 2π rad)")
plt.title("Firing Rates across 2 Cycles of Vertical Gratings")
plt.show()

#Spike rate as a response to oriented grating images
theta = np.arange(0,2*np.pi,0.01)
responses = [0 for i in range(len(alphas))]
for i in range(len(theta)):
    Io = np.sin(k * (x * np.cos(theta[i]) - y * np.sin(theta[i])))
    RFtimesImage = np.dot(RF.flatten(), Io.flatten())    
    hw_nonlinearity = max(0, RFtimesImage)**2
    responses[i] = hw_nonlinearity

responses = peak_rate * (responses / np.max(responses))

plt.imshow(np.sin(k * (x * np.cos(1.57) - y * np.sin(1.57))), cmap='gray', extent=[-5,5,-5,5])
plt.title('Vertical Grating, Orientation = 90 deg')
plt.xlabel("x (deg)")
plt.ylabel("y (deg)")
plt.show()

plt.plot(theta, responses)
plt.ylabel("Firing Rate (spikes/second)")
plt.xlabel("Grating Orientation (0 to 2π rad)")
plt.title("Firing Rate across 2π rad Orientation Rotations")
plt.show()

n_images = 100000
image_shape = RF.shape
wn_images = np.random.randn(n_images, image_shape[0]*image_shape[1])

linear = wn_images @ RF.flatten()  
nonlinear = np.maximum(0, linear)**2


mean = np.mean(nonlinear)
scale = 0.2 / mean
firing_rates = nonlinear * scale
spike_counts = np.random.poisson(firing_rates)

print(f"Mean scaled firing rate = {np.mean(firing_rates):.4f}")
print(f"Mean spike count across all images = {np.mean(spike_counts):.4f}")

total_spikes = np.sum(spike_counts)

if total_spikes > 0:
    #Average over all images that generated at least 1 spike
    STA_V = (spike_counts[:, None] * wn_images).sum(axis=0) / total_spikes
    STA = STA_V.reshape(image_shape)

    #True RF vs. STA
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(RF, cmap='bwr', extent=[-5,5,-5,5])
    plt.title("True RF")
    plt.xlabel("x (deg)")
    plt.ylabel("y (deg)")
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(STA, cmap='bwr', extent=[-5,5,-5,5])
    plt.xlabel("x (deg)")
    plt.ylabel("y (deg)")
    plt.title("Spike-Triggered Average (STA)")
    plt.colorbar()
    plt.show()

    #RF correlaton coefficient calculation
    corr = np.corrcoef(RF.flatten(), STA_V)[0,1]
    print(f"Correlation between RF and STA: {corr:.3f}")
else:
    print("No images produced a spike, check RF")

subset_sizes = [100, 1000, 10000, 100000]
STA_images = []
correlations = []

for N in subset_sizes:
    sub_imgs = wn_images[:N]
    sub_spikes = spike_counts[:N]
    total_spikes = np.sum(sub_spikes)
    if total_spikes == 0:
        STA = np.zeros(image_shape)
    else:
        STA_V = (sub_spikes[:, None] * sub_imgs).sum(axis=0) / total_spikes
        STA = STA_V.reshape(image_shape)
    STA_images.append(STA)

    #RF correlation coefficient calculation
    corr = np.corrcoef(RF.flatten(), STA.flatten())[0, 1]
    correlations.append(corr)

#STA vs true RF plots
fig, axs = plt.subplots(1, 5, figsize=(20, 4))
axs[0].imshow(RF, cmap='bwr', extent=[-5,5,-5,5])
axs[0].set_xlabel("x (deg)")
axs[0].set_ylabel("y (deg)")
axs[0].set_title('True RF')

for i, N in enumerate(subset_sizes):
    axs[i+1].imshow(STA_images[i], cmap='bwr', extent=[-5,5,-5,5])
    axs[i+1].set_xlabel("x (deg)")
    axs[i+1].set_title(f'STA (N={N})')
plt.show()

#Correlation vs. num white noise images (log scale)
plt.figure()
plt.plot(subset_sizes, correlations, marker='o')
plt.xscale('log')
plt.xlabel('Number of white noise images (log scale)')
plt.ylabel('Correlation (STA vs RF)')
plt.title('Convergence of STA to true RF')
plt.show()