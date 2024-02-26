import numpy as np
from scipy.spatial.distance import cdist
from skimage import io, color
from skimage.util import view_as_windows

def find_best_match(patch, texture, patch_size):
    """
    Find the best matching patch in the texture for the given patch.
    """
    texture_patches = view_as_windows(texture, (patch_size, patch_size))
    texture_patches = texture_patches.reshape(-1, patch_size * patch_size)
    patch = patch.flatten()
    distances = cdist([patch], texture_patches, 'euclidean')
    min_dist_idx = np.argmin(distances)
    return np.unravel_index(min_dist_idx, texture.shape[:2])

def texture_synthesis(source_texture, target_size, patch_size=9, seed_size=3):
    """
    Synthesize a new texture from a source texture using Efros and Leung's algorithm.
    """
    source_texture = color.rgb2gray(source_texture)  # Convert to grayscale
    target_texture = np.zeros(target_size)  # Initialize target texture
    seed_pos = (target_size[0]//2, target_size[1]//2)
    
    # Plant the seed from the source texture into the center of the target texture
    seed = source_texture[:seed_size, :seed_size]
    start_pos = (seed_pos[0]-seed_size//2, seed_pos[1]-seed_size//2)
    target_texture[start_pos[0]:start_pos[0]+seed_size, start_pos[1]:start_pos[1]+seed_size] = seed
    
    # Pad source texture for border handling
    pad_width = patch_size // 2
    padded_texture = np.pad(source_texture, pad_width, mode='reflect')
    
    # Synthesize texture
    for i in range(target_size[0]):
        for j in range(target_size[1]):
            if target_texture[i, j] > 0:  # Skip already filled pixels
                continue
            # Define the neighborhood window
            i_min, i_max = max(i - pad_width, 0), min(i + pad_width + 1, target_size[0])
            j_min, j_max = max(j - pad_width, 0), min(j + pad_width + 1, target_size[1])
            window = target_texture[i_min:i_max, j_min:j_max]
            
            # Find the best matching patch in the source texture
            best_match_pos = find_best_match(window, padded_texture, patch_size)
            # Copy the pixel value from the center of the best matching patch
            target_texture[i, j] = padded_texture[best_match_pos[0] + pad_width, best_match_pos[1] + pad_width]
    
    return target_texture

# Example usage
source_texture = io.imread('textur1.jpeg')  # Load your source texture
source_texture = source_texture.resize(50,50,3)
target_size = (100, 100)  # Desired size of the synthesized texture
synthesized_texture = texture_synthesis(source_texture, target_size, patch_size=9, seed_size=3)
io.imsave('synthesized_texture.jpg', synthesized_texture)  # Save the synthesized texture
