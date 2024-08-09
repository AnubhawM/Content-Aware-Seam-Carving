import os
import numpy as np
import cv2
import numba
import scipy.ndimage as sp
from copy import deepcopy

# Generate an energy map for an image using backward energy
@numba.jit
def calculate_backward_energy(image):

    # Ensure the image is of dtype np.float32
    image = image.astype(np.float32)

    # Accounting for the partial derivative with respect to x
    dx_filter = np.asarray([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0]
    ])

    # Accounting for the partial derivative with respect to y
    dy_filter = np.asarray([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0]
    ])

    # Convert dx_filter into a 3D filter since the input is a color image
    dx_filter = np.stack([dx_filter] * 3, axis=2)

    # Convert dy_filter into a 3D filter since the input is a color image
    dy_filter = np.stack([dy_filter] * 3, axis=2)

    # Apply the filters to the image in the respective directions
    dx = sp.convolve(image, dx_filter)
    dy = sp.convolve(image, dy_filter)

    # This is the given equation for energy function e1 in the 2007 paper
    e1 = np.sum(np.absolute(dx) + np.absolute(dy), axis=2)

    return e1


@numba.jit
def calculate_forward_energy(image):

    rows = image.shape[0]
    cols = image.shape[1]
    
    # Convert the image to grayscale
    I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure the image is of dtype np.float64
    I = I.astype(np.float64)

    # Initialize array for forward energy
    forward_energy = np.zeros((rows, cols))

    # Initialize array for cumulative cost matrix M
    M = np.zeros_like(forward_energy)
    
    for i in range(1, rows):

        for j in range(cols):
            
            # Equations from the paper for calculating costs
            cL = np.abs(I[i, j+1] - I[i, j-1]) + np.abs(I[i-1, j] - I[i, j-1])
            cU = np.abs(I[i, j+1] - I[i, j-1])
            cR = np.abs(I[i, j+1] - I[i, j-1]) + np.abs(I[i-1, j] - I[i, j+1])

            choices = np.array( [M[i-1, j-1] + cL, 
                                 M[i-1, j] + cU, 
                                 M[i-1, j+1] + cR] )         

            # Determine which cost is the lowest
            minimum = np.argmin(choices)

            # Update the new accumulative cost matrix, M, with the minimum cost
            M[i, j] = choices[minimum]

            # Accumulate the individual costs into an array to be able to access
            # them by index
            choices_costs = np.array([cL, cU, cR])

            # Choose the cost which led to the lowest accumulative cost when
            # added with the corresponding value in the accumulative cost
            # matrix M
            forward_energy[i,j] = choices_costs[minimum]

            
    return forward_energy

# Calculate the minimum seam in an image and
# generate an array for backtracking consisting
# of the optimal seam pixel indices
@numba.jit
def calculate_minimum_seam(image, energy_type):
    
    energy_map = None

    if energy_type == "backward":
        # Get the image's energy map using backward energy
        energy_map = calculate_backward_energy(image)

    elif energy_type == "forward":
        # Get the image's energy map using forward energy
        energy_map = calculate_forward_energy(image)

    # Initialize an array for the cumulative minimum energy, M
    M = np.zeros_like(energy_map)

    # Copy the the first row from energy_map to the first row of 
    # M because the cumulative minimum energy for the first row of values
    # are the values themselves
    M[0, :] = energy_map[0, :]

    # Initialize an array to store the indices of the pixels which are
    # part of the minimum seam
    # This will be used for backtracking later to find the minimum seam
    optimal_seam_pixel_indices = np.zeros_like(M, np.int)

    rows = image.shape[0]
    cols = image.shape[1]

    minimum_seam = np.zeros(rows)

    # Start at the second row
    for row in range(1, rows):
        # Start from the first column
        for col in range(cols):
            # Since starting at the left-most column,
            # need to handle the left edge indices
            if col == 0:

                # row-1 is the previous row
                # 0:2 means start at the current column(leftmost edge) and
                # go one spot forward. This maintains the 8-connected definition
                # of a vertical seam
                idx = np.argmin(M[row-1, 0:2])

                # Calculate the cumulative minimum energy for the seam
                # by adding the energy at that point from the energy_map
                # with the minimum energy value from the previous row in M
                M[row, 0] = energy_map[row, 0] + M[row-1, idx]

                # Need to remember the index of the pixel in the minimum seam
                # to perform backtracking later
                optimal_seam_pixel_indices[row, 0] = idx

            else:

                # The col-1 is for reaching up until the current column
                # (the col is because indexing starts at 0)
                # idx is in the range [0, 2]                
                idx = np.argmin(M[row-1, col-1:col+2])

                # Calculate the cumulative minimum energy for the seam
                # by adding the energy at that point from the energy_map
                # with the minimum energy value from the previous row in M
                M[row, col] = energy_map[row, col] + M[row-1, idx + col-1]

                # Need to remember the index of the pixel in the minimum seam
                # to perform backtracking later
                optimal_seam_pixel_indices[row, col] = idx + col - 1


    ## Backtracking to construct the minimum seam ##

    # Get the index of the minimum value of the last row in M - this is 
    # the minimum entry from which backtracking can begin
    col = np.argmin(M[-1])

    # Traverse backwards
    for row in range(rows-1, -1, -1):

        minimum_seam[row] = optimal_seam_pixel_indices[row, col]
        
        # Update col to hold the index of the next pixel which
        # is part of the seam and one row up by accessing that
        # value from the optimal_seam_pixel_indices array
        # optimal_seam_pixel_indices holds the index where the current
        # pixel value came from, i.e., it holds the index of the next pixel
        # which is part of the minimum seam one row up        
        col = optimal_seam_pixel_indices[row, col]
    
    return minimum_seam

@numba.jit
def remove_seam_helper(image, scale, energy_type):
    
    image_copy = deepcopy(image)
    
    rows = image_copy.shape[0]
    cols = image_copy.shape[1]

    # Compute the modified scale to convert the image into 
    modified_scale = int(cols * scale)

    # Compute the number of seams to remove
    num_seams_to_remove = cols - modified_scale

    for _ in range(num_seams_to_remove):

        # Calculate energy map based on the the energy_type parameter
        # It will either be by backward energy or forward energy
        minimum_seam = calculate_minimum_seam(image_copy, energy_type)

        # Remove the seam
        image_copy = remove_seam(image_copy, minimum_seam)
    
    return image_copy


# Take the minimum seam out of the image
@numba.jit
def remove_seam(image, minimum_seam, to_draw=False):
    image_copy = deepcopy(image)
    reduced_image = deepcopy(image)
    rows = image.shape[0]
    cols = image.shape[1]
    
    # Create a boolean mask
    mask =  np.ones_like(image, dtype=bool)
    for row in range(rows):

        if to_draw:
            # Make the pixel red
            reduced_image[row, int(minimum_seam[row])] = np.array([0, 0, 255])
        else:    
            mask[row, int(minimum_seam[row])] = False

    if not to_draw:
        reduced_image = image_copy[mask].reshape((rows, cols-1, 3))

    return reduced_image


@numba.jit
def  insert_seam_helper(image, num_seams_to_insert, energy_type, to_draw=False):
    
    image_copy = deepcopy(image)

    # Store the seams to insert in the order they are removed from the seam removal process
    seams_to_insert = []

    for _ in range(num_seams_to_insert):
        
        # Calculate the minimum seam
        minimum_seam = calculate_minimum_seam(image_copy, energy_type)
        
        # Add the minimum seam to the seam store
        seams_to_insert.append(minimum_seam)

        # Remove the seam using seam removal
        image_copy = remove_seam(image_copy, minimum_seam, to_draw)

    # Make a copy of the passed-in, unmodified image
    image_to_expand = deepcopy(image)
    
    # Perform the seam insertion seam by seam
    for _ in range(num_seams_to_insert):

        # Retrieve the minimum seam from the seams_to_insert list in the order that it
        # was removed from the image by the seam removal process AND remove it from the seam store
        minimum_seam = seams_to_insert.pop(0)

        # Perform the seam insertion
        image_to_expand = insert_seam(image_to_expand, minimum_seam, to_draw)

        # Update the seams in the seams_to_insert list by shifting them
        # to account for the offset image size after having inserted a seam
        for seam in seams_to_insert:

            for idx, val in enumerate(seam):

                # If index of current seam is after the current
                # minimum seam, then it needs to be offset
                if val >= minimum_seam[idx]:

                    seam[idx] += 2

    return image_to_expand


@numba.jit
def insert_seam(image, seam, to_draw):

    rows = image.shape[0]
    cols = image.shape[1]
    
    # Account for a new seam being inserted by creating
    # a new image with 1 extra column
    expanded_image = np.zeros((rows, cols + 1, 3))

    # If need to draw red seams on the image
    if to_draw:

        for row in range(rows):

            # MAKE SURE THIS IS AN INT
            col = int(seam[row])

            for channel in range(3):

                if col == 0:

                    expanded_image[row, col, channel] = image[row, col, channel]
                    
                    # Modify the color of the pixel to be red
                    if channel == 0:
                        expanded_image[row, col, channel] = 0
                    elif channel == 1:
                        expanded_image[row, col, channel] = 0
                    elif channel == 2:
                        expanded_image[row, col, channel] = 255

                    expanded_image[row, col+1:, channel] = image[row, col:, channel]

                else:

                    expanded_image[row, :col, channel] = image[row, :col, channel]

                    if channel == 0:
                        expanded_image[row, col, channel] = 0
                    elif channel == 1:
                        expanded_image[row, col, channel] = 0
                    elif channel == 2:
                        expanded_image[row, col, channel] = 255

                    expanded_image[row, col+1:, channel] = image[row, col:, channel]

    else:

        for row in range(rows):

            # MAKE SURE THIS IS AN INT
            col = int(seam[row])

            for channel in range(3):

                if col == 0:

                    expanded_image[row, col, channel] = image[row, col, channel]

                    # Calculate average of the columns directly to the left and
                    # to the right of the current column
                    left = image[row, col, channel]
                    right = image[row, col+2, channel]
                    avg = (left + right) / 2

                    # Add the seam pixel
                    expanded_image[row, col+1, channel] = avg
                    
                    # Copy the rest of the image after the seam pixel location to the expanded image
                    expanded_image[row, col+1:, channel] = image[row, col:, channel]

                else:

                    # Copy everything from the original image to the expanded image up
                    # until the location where the seam needs to be added
                    expanded_image[row, :col, channel] = image[row, :col, channel]

                    # Calculate average of the columns directly to the right and
                    # to the left of the current column
                    left = image[row, col-1, channel]
                    right = image[row, col+2, channel]
                    avg = (left + right) / 2

                    # Add the seam
                    expanded_image[row, col, channel] = avg

                    # Copy the rest of the image after the seam pixel location to the expanded image
                    expanded_image[row, col+1:, channel] = image[row, col:, channel]

    return expanded_image


# Same as remove_seam_helper, but for drawing seams
def draw_seams_on_insertion(image_to_draw_seams_on, scale, energy_type):

    image_copy = deepcopy(image_to_draw_seams_on)

    rows = image_copy.shape[0]
    cols = image_copy.shape[1]

    # Compute the modified scale to convert the image into 
    modified_scale = int(cols * scale)

    # Compute the number of seams to remove
    num_seams_to_remove = cols - modified_scale

    for _ in range(num_seams_to_remove):

        # Calculate energy map based on the the energy_type parameter
        minimum_seam = calculate_minimum_seam(image_copy, energy_type)

        # Remove the seam
        image_copy = remove_seam(image_copy, minimum_seam, True)
    
    return image_copy

def draw_seams(image_to_draw_seams_on, num_seams_to_insert, energy_type):

    image_copy = deepcopy(image_to_draw_seams_on)

    # Store the seams to insert in the order they are removed from the seam removal process
    seams_to_insert = []

    for _ in range(num_seams_to_insert):
        
        # Calculate the minimum seam
        minimum_seam = calculate_minimum_seam(image_copy, energy_type)
        
        # Add the minimum seam to the seam store
        seams_to_insert.append(minimum_seam)

        # Remove the seam using seam removal
        image_copy = remove_seam(image_copy, minimum_seam)

    # Make a copy of the passed-in, unmodified image
    image_to_expand = deepcopy(image_to_draw_seams_on)
    
    # Perform the seam insertion seam by seam
    for _ in range(num_seams_to_insert):
        
        # Retrieve the minimum seam from the seams_to_insert list in the order that it
        # was removed from the image by the seam removal process AND remove it from the seam store
        minimum_seam = seams_to_insert.pop(0)

        # Perform the seam insertion
        image_to_expand = insert_seam(image_to_expand, minimum_seam, True)

        # Update the seams in the seams_to_insert list by shifting them
        # to account for the offset image size after having inserted a seam
        for seam in seams_to_insert:

            for idx, val in enumerate(seam):

                # If index of current seam is after the current
                # minimum seam, then it needs to be offset
                if val >= minimum_seam[idx]:

                    seam[idx] += 2

    return image_to_expand


if __name__ == "__main__":

    SOURCE_IMAGES = "source images/base/"
    COMPARISON_IMAGES = "source images/comparison/"
    RESULTS_FOLDER = "results/"

    subfolders = os.walk(SOURCE_IMAGES)

    files = []
    for dirpath, _, filenames in subfolders:
        for file in filenames:
            files.append(file)

    for source_image in files:
        
        # Waterfall 50% seam removal
        if source_image == "fig5_07_base.png":
            image = cv2.imread(dirpath+source_image)
            image_copy = deepcopy(image)
            output_image = remove_seam_helper(image_copy, 0.5, "backward")
            cv2.imwrite(RESULTS_FOLDER+"fig5_07_result.png", output_image)

        # Dolphin
        elif source_image == "fig8_07_base.png":

            image = cv2.imread(dirpath+source_image)

            image_copy = deepcopy(image)

            num_seams_to_insert = 113

            dolphin_with_seams = draw_seams(image_copy, num_seams_to_insert, "backward")
            cv2.imwrite(RESULTS_FOLDER+"fig8c_07_result.png", dolphin_with_seams)

            # Dolphin 50% seam insertion
            image_copy = deepcopy(image)

            num_seams_to_insert = 119
            output_image = insert_seam_helper(image_copy, num_seams_to_insert, "backward")
            cv2.imwrite(RESULTS_FOLDER+"fig8d_07_result.png", output_image)

            # Dolphin with 2 steps of 50% seam insertion
            image_copy = deepcopy(output_image)

            num_seams_to_insert = 121
            output_image = insert_seam_helper(image_copy, num_seams_to_insert, "backward")
            cv2.imwrite(RESULTS_FOLDER+"fig8f_07_result.png", output_image)
        
        # Bench
        elif source_image == "fig8_08_base.png":

            image = cv2.imread(dirpath+source_image)

            # Bench with red seams to be removed by backward energy
            image_copy = deepcopy(image)
            image_with_seams_drawn = draw_seams_on_insertion(image_copy, 0.5, "backward")
            cv2.imwrite(RESULTS_FOLDER+"fig8_08_back_seam_result.png", image_with_seams_drawn)

            # Bench with seams removed using backward energy
            
            image_copy = deepcopy(image)
            output_image = remove_seam_helper(image_copy, 0.5, "backward")
            cv2.imwrite(RESULTS_FOLDER+"fig8_08_backward_result.png", output_image)

            # Bench with red seams to be removed by forward energy
            image_copy = deepcopy(image)
            image_with_seams_drawn = draw_seams_on_insertion(image_copy, 0.5, "forward")
            cv2.imwrite(RESULTS_FOLDER+"fig8_08_forward_seam_result.png", image_with_seams_drawn)

            # Bench with seams removed using forward energy
            image_copy = deepcopy(image)
            output_image = remove_seam_helper(image_copy, 0.5, "forward")
            cv2.imwrite(RESULTS_FOLDER+"fig8_08_forward_result.png", output_image)
        

        # Car
        elif source_image == "fig9_08_base.png":
            
            image = cv2.imread(dirpath+source_image)
            # Stretched car with seams inserted using backward energy
            
            image_copy = deepcopy(image)
            cols = image_copy.shape[1]
            modified_scale = int(cols * 0.5)
            num_seams_to_insert = cols - modified_scale
            output_image = insert_seam_helper(image_copy, num_seams_to_insert, "backward")
            cv2.imwrite(RESULTS_FOLDER+"fig9_08_backward_result.png", output_image)

            # Stretched car with seams inserted using forward energy

            image_copy = deepcopy(image)
            cols = image_copy.shape[1]
            modified_scale = int(cols * 0.5)
            num_seams_to_insert = cols - modified_scale
            output_image = insert_seam_helper(image_copy, num_seams_to_insert, "forward")
            cv2.imwrite(RESULTS_FOLDER+"fig9_08_forward_result.png", output_image)



    # Metrics for waterfall

    print("\nFor waterfall: \n")
    image_mine = cv2.imread(RESULTS_FOLDER+"fig5_07_result.png")
    image_base = cv2.imread(SOURCE_IMAGES+"fig5_07_base.png")
    image_comparison = cv2.imread(COMPARISON_IMAGES+"fig5_07_seam_removal.png")

    energy_mine = calculate_backward_energy(image_mine)
    energy_base = calculate_backward_energy(image_base)
    energy_comparison = calculate_backward_energy(image_comparison)

    sum_mine = np.sum(energy_mine)
    sum_base = np.sum(energy_base)
    sum_comparison = np.sum(energy_comparison)
    
    print("Energy ratio of base to my result: ", sum_base/sum_mine)
    print("Energy ratio of my result to comparison: ", sum_mine/sum_comparison)
    print("Energy ratio of base to comparison: ", sum_base/sum_comparison)


    print("\nFor dolphin 8d: \n")

    # Metrics for Dolphin 8d

    image_mine = cv2.imread(RESULTS_FOLDER+"fig8d_07_result.png")
    image_base = cv2.imread(SOURCE_IMAGES+"fig8_07_base.png")
    image_comparison = cv2.imread(COMPARISON_IMAGES+"fig8d_07_insert50.png")

    energy_mine = calculate_backward_energy(image_mine)
    energy_base = calculate_backward_energy(image_base)
    energy_comparison = calculate_backward_energy(image_comparison)

    sum_mine = np.sum(energy_mine)
    sum_base = np.sum(energy_base)
    sum_comparison = np.sum(energy_comparison)
    
    print("Energy ratio of base to my result: ", sum_base/sum_mine)
    print("Energy ratio of my result to comparison: ", sum_mine/sum_comparison)
    print("Energy ratio of base to comparison: ", sum_base/sum_comparison)

    print("\nFor dolphin 8f: \n")

    # Metrics for Dolphin 8f

    image_mine = cv2.imread(RESULTS_FOLDER+"fig8f_07_result.png")
    image_base = cv2.imread(SOURCE_IMAGES+"fig8_07_base.png")
    image_comparison = cv2.imread(COMPARISON_IMAGES+"fig8f_insert50-50.png")

    energy_mine = calculate_backward_energy(image_mine)
    energy_base = calculate_backward_energy(image_base)
    energy_comparison = calculate_backward_energy(image_comparison)

    sum_mine = np.sum(energy_mine)
    sum_base = np.sum(energy_base)
    sum_comparison = np.sum(energy_comparison)
    
    print("Energy ratio of base to my result: ", sum_base/sum_mine)
    print("Energy ratio of my result to comparison: ", sum_mine/sum_comparison)
    print("Energy ratio of base to comparison: ", sum_base/sum_comparison)


    print("\nFor bench seams removed by backward energy: \n")

    # Metrics for Bench with seams removed using backward energy

    image_mine = cv2.imread(RESULTS_FOLDER+"fig8_08_backward_result.png")
    image_base = cv2.imread(SOURCE_IMAGES+"fig8_08_base.png")
    image_comparison = cv2.imread(COMPARISON_IMAGES+"fig8_08_backward_energy.png")

    energy_mine = calculate_backward_energy(image_mine)
    energy_base = calculate_backward_energy(image_base)
    energy_comparison = calculate_backward_energy(image_comparison)

    sum_mine = np.sum(energy_mine)
    sum_base = np.sum(energy_base)
    sum_comparison = np.sum(energy_comparison)
    
    print("Energy ratio of base to my result: ", sum_base/sum_mine)
    print("Energy ratio of my result to comparison: ", sum_mine/sum_comparison)
    print("Energy ratio of base to comparison: ", sum_base/sum_comparison)

    print("\nFor bench with seams removed by forward energy: \n")

    # Metrics for Bench with seams removed using forward energy

    image_mine = cv2.imread(RESULTS_FOLDER+"fig8_08_forward_result.png")
    image_base = cv2.imread(SOURCE_IMAGES+"fig8_08_base.png")
    image_comparison = cv2.imread(COMPARISON_IMAGES+"fig8_08_ forward_energy.png")

    energy_mine = calculate_backward_energy(image_mine)
    energy_base = calculate_backward_energy(image_base)
    energy_comparison = calculate_backward_energy(image_comparison)

    sum_mine = np.sum(energy_mine)
    sum_base = np.sum(energy_base)
    sum_comparison = np.sum(energy_comparison)
    
    print("Energy ratio of base to my result: ", sum_base/sum_mine)
    print("Energy ratio of my result to comparison: ", sum_mine/sum_comparison)
    print("Energy ratio of base to comparison: ", sum_base/sum_comparison)   

    print("\nFor stretched car with seams inserted using backward energy: \n")

    # Metrics for stretched car with seams inserted using backward energy

    image_mine = cv2.imread(RESULTS_FOLDER+"fig9_08_backward_result.png")
    image_base = cv2.imread(SOURCE_IMAGES+"fig9_08_base.png")
    image_comparison = cv2.imread(COMPARISON_IMAGES+"fig9_08_backward_energy.png")

    energy_mine = calculate_backward_energy(image_mine)
    energy_base = calculate_backward_energy(image_base)
    energy_comparison = calculate_backward_energy(image_comparison)

    sum_mine = np.sum(energy_mine)
    sum_base = np.sum(energy_base)
    sum_comparison = np.sum(energy_comparison)
    
    print("Energy ratio of base to my result: ", sum_base/sum_mine)
    print("Energy ratio of my result to comparison: ", sum_mine/sum_comparison)
    print("Energy ratio of base to comparison: ", sum_base/sum_comparison)   

    print("\nFor stretched car with seams inserted using forward energy\n")

    # Metrics for stretched car with seams inserted using forward energy

    image_mine = cv2.imread(RESULTS_FOLDER+"fig9_08_forward_result.png")
    image_base = cv2.imread(SOURCE_IMAGES+"fig9_08_base.png")
    image_comparison = cv2.imread(COMPARISON_IMAGES+"fig9_08_forward_energy.png")

    energy_mine = calculate_backward_energy(image_mine)
    energy_base = calculate_backward_energy(image_base)
    energy_comparison = calculate_backward_energy(image_comparison)

    sum_mine = np.sum(energy_mine)
    sum_base = np.sum(energy_base)
    sum_comparison = np.sum(energy_comparison)
    
    print("Energy ratio of base to my result: ", sum_base/sum_mine)
    print("Energy ratio of my result to comparison: ", sum_mine/sum_comparison)
    print("Energy ratio of base to comparison: ", sum_base/sum_comparison)   
    
    print()



