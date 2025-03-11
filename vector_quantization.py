import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import os
from sklearn.cluster import KMeans

def load_image_as_gray(image_path):
    """
    Load an image from file and convert it to grayscale [0,1].
    """
    img = io.imread(image_path)
    
    # If there's an alpha channel, remove it:
    if img.ndim == 3 and img.shape[-1] == 4:
        # Convert RGBA to RGB
        img = color.rgba2rgb(img)
    
    # Convert RGB or RGBA to grayscale
    if len(img.shape) == 3:
        img = color.rgb2gray(img)
    
    # Ensure floating-point with range [0,1] if possible
    img = img.astype(np.float64)
    return img

def extract_blocks(img, block_size=(4, 4)):
    """
    Extract non-overlapping blocks of size block_size from the image.
    Returns:
      blocks: 2D array of shape (num_blocks, block_size[0]*block_size[1])
      shape_info: (nR, nC) block layout for reconstruction
    """
    rows, cols = img.shape
    br, bc = block_size
    
    # Number of 4x4 blocks that fit in the image
    nR = rows // br
    nC = cols // bc
    img_cropped = img[:nR*br, :nC*bc]
    
    blocks = []
    for r in range(nR):
        for c in range(nC):
            block = img_cropped[r*br:(r+1)*br, c*bc:(c+1)*bc]
            blocks.append(block.flatten())
    
    return np.array(blocks), (nR, nC)

def reconstruct_image_from_blocks(blocks, shape_info, block_size=(4, 4)):
    """
    Reconstruct an image from flattened blocks.
    """
    nR, nC = shape_info
    br, bc = block_size
    recon_img = np.zeros((nR*br, nC*bc))
    
    idx = 0
    for r in range(nR):
        for c in range(nC):
            block_flat = blocks[idx]
            block_2d = block_flat.reshape(br, bc)
            recon_img[r*br:(r+1)*br, c*bc:(c+1)*bc] = block_2d
            idx += 1
    
    return recon_img

def train_codebook(blocks, num_centroids):
    """
    Train a codebook using K-Means (similar to Generalized Lloyd algorithm).
    """
    kmeans = KMeans(n_clusters=num_centroids, n_init=10, random_state=42)
    kmeans.fit(blocks)
    return kmeans

def quantize_blocks(blocks, kmeans):
    """
    Quantize blocks by assigning each block to its closest centroid.
    """
    labels = kmeans.predict(blocks)
    centroids = kmeans.cluster_centers_
    quantized_blocks = centroids[labels]
    return quantized_blocks

def psnr(original, reconstructed):
    """
    Compute Peak Signal-to-Noise Ratio between original and reconstructed.
    Assumes both in the same range (e.g., [0,1]).
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100  
    max_val = np.max(original)
    return 20 * np.log10(max_val / np.sqrt(mse))

def main():
    # 1) Example single-image training
    image_path = "sample_image/barbara.png"
    img_gray = load_image_as_gray(image_path)

    # Extract 4x4 blocks
    blocks, shape_info = extract_blocks(img_gray, block_size=(4, 4))

    # Train codebook for L=128
    num_centroids = 128
    kmeans_128 = train_codebook(blocks, num_centroids)
    quantized_blocks_128 = quantize_blocks(blocks, kmeans_128)
    recon_img_128 = reconstruct_image_from_blocks(quantized_blocks_128, shape_info, (4, 4))

    # Compute PSNR
    psnr_128 = psnr(img_gray[:recon_img_128.shape[0], :recon_img_128.shape[1]], recon_img_128)
    print(f"PSNR for L=128 codebook: {psnr_128:.2f} dB")

    # Train codebook for L=256
    num_centroids = 256
    kmeans_256 = train_codebook(blocks, num_centroids)
    quantized_blocks_256 = quantize_blocks(blocks, kmeans_256)
    recon_img_256 = reconstruct_image_from_blocks(quantized_blocks_256, shape_info, (4, 4))

    psnr_256 = psnr(img_gray[:recon_img_256.shape[0], :recon_img_256.shape[1]], recon_img_256)
    print(f"PSNR for L=256 codebook: {psnr_256:.2f} dB")

    # 2) Training on multiple images
    folder_path = "sample_image"
    image_list = [
        "airplane.png",
        "barbara.png",
        "boat.png",
        "cat.png",
        "fruits.png",
        "girl.png",
        "peppers.png",
        "tulips.png",
        "watch.png",
        "zelda.png"
    ]

    # Collect all blocks from multiple images
    all_blocks = []
    for fname in image_list:
        multi_img_gray = load_image_as_gray(os.path.join(folder_path, fname))
        b, _ = extract_blocks(multi_img_gray, block_size=(4, 4))
        all_blocks.append(b)
    all_blocks = np.vstack(all_blocks)

    # Train a codebook on the combined blocks
    num_centroids_multi = 128
    kmeans_multi = train_codebook(all_blocks, num_centroids_multi)

    quantized_blocks_multi = quantize_blocks(blocks, kmeans_multi)
    recon_img_multi = reconstruct_image_from_blocks(quantized_blocks_multi, shape_info, (4, 4))

    psnr_multi = psnr(img_gray[:recon_img_multi.shape[0], :recon_img_multi.shape[1]], recon_img_multi)
    print(f"PSNR using multi-image codebook: {psnr_multi:.2f} dB")

    # Display results
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    axs[0].imshow(img_gray, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(recon_img_128, cmap='gray')
    axs[1].set_title(f"L=128 (PSNR={psnr_128:.2f} dB)")
    axs[1].axis('off')

    axs[2].imshow(recon_img_256, cmap='gray')
    axs[2].set_title(f"L=256 (PSNR={psnr_256:.2f} dB)")
    axs[2].axis('off')

    axs[3].imshow(recon_img_multi, cmap='gray')
    axs[3].set_title(f"Multi-image CB (PSNR={psnr_multi:.2f} dB)")
    axs[3].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
