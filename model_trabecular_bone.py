import sys
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def rotate_clockwise(image, degree):
    """ Function rotating an image clockwise. """

    # Get the image size
    (height, width) = image.shape[:2]

    # Define the center of the image
    center = (width / 2, height / 2)

    # Perform the general rotation
    angle = degree
    scale = 1.0
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_by_matrix = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_NEAREST)

    return rotated_by_matrix


def rotate_counterclockwise(image, degree):
    """ Function rotating an image counterclockwise. """

    # Get the image size
    (height, width) = image.shape[:2]

    # Define the center of the image
    center = (width / 2, height / 2)

    # Perform the general rotation
    angle = -degree
    scale = 1.0
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_by_matrix = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_NEAREST)

    return rotated_by_matrix

def generate_white_noise_map(map_size):
    """Create a 2D array of white noise with size
    map_size x map_size."""

    return np.random.random(size=(map_size, map_size))


def combine_gaussian_envelope_and_noise(map_size, sigma, white_noise):
    """ Create a Gaussian enveloping function and combine it with the white noise """

    # Create a grid of coordinates
    j = np.arange(map_size)
    k = np.arange(map_size - 1, -1, -1)
    jj, kk = np.meshgrid(j, k, indexing='xy')

    # Define the enveloping function
    enveloping_function = np.exp(-(jj ** 2 + kk ** 2) / (2 * sigma ** 2))

    # Point-wise product of white noise and Gaussian enveloping function
    point_wise_product = np.multiply(enveloping_function, white_noise)

    return point_wise_product


def combine_lorentzian_envelope_and_noise(map_size, sigma, white_noise, j0, k0):
    """ Create a Lorentzian enveloping function and combine it with the white noise """

    # Create a grid of coordinates
    j = np.arange(map_size)
    k = np.arange(map_size - 1, -1, -1)
    jj, kk = np.meshgrid(j, k, indexing='xy')

    # Define the enveloping function
    enveloping_function = (sigma / ((jj - j0) ** 2 + sigma ** 2)) * (sigma / ((kk - k0) ** 2 + sigma ** 2))

    # Point-wise product of white noise and Lorentzian enveloping function
    point_wise_product = np.multiply(enveloping_function, white_noise)

    return point_wise_product


def fourier_transform(point_wise_product):
    """ Fourier transform along horizontal and transverse pixel coordinates """

    # Do Fourier transform
    fourier_transform = np.fft.fft2(point_wise_product)

    # Scale the Fourier transform to have unit modulus
    scaled_fourier_transform = fourier_transform / np.abs(fourier_transform)

    return scaled_fourier_transform


def R_concentric(map_size, a, b, q0, r0, L):
    """ Create complex function R for the concentric structures """

    # Create a grid of coordinates
    q = np.arange(map_size)
    r = np.arange(map_size - 1, -1, -1)
    qq, rr = np.meshgrid(q, r, indexing='xy')

    # Complex function R
    R = np.exp(-1j * (((a * (qq - q0) ** 2 + b * (rr - r0) ** 2) ** 0.8) / L))

    return R


def R_nonconcentric(map_size):
    """ Create complex function R for the non-concentric structures """

    # Create a grid of coordinates
    q = np.arange(map_size)
    r = np.arange(map_size - 1, -1, -1)
    qq, rr = np.meshgrid(q, r, indexing='xy')

    # Complex function R
    R = np.exp(-1j * (qq + rr) ** 0.1)

    return R


def resulting_function_T(R, scaled_fourier_transform, eta):
    """ Function T combining R and the fourier transform. The two complex fields are combined and converted back to real space by raising the modulus to a factor > 2 in order to reduce the resulting wall thickness in a tunable manner. """

    # Function T
    T = np.abs(scaled_fourier_transform + R) ** (
            2 * eta)

    # Normalize T so the values fall within 0 and 255
    T_normalized = cv2.normalize(T, None, 0, 255, cv2.NORM_MINMAX)
    T_normalized = T_normalized.astype(np.uint8)

    # Use OTSU threshold to binarize the structures
    ret2, th2 = cv2.threshold(T_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return th2

def parameters(bone):
    if bone == 'patella':
        sigma1 = 70
        eta1 = 6.4
        j01 = 50
        k01 = 0
        sigma2 = 100
        eta2 = 3.5
        j02 = 25
        k02 = 25
        d = 7

        return sigma1, eta1, j01, k01, sigma2, eta2, j02, k02, d

    elif bone == 'proximal fibula':
        sigma = 100
        L = 400
        a = 20
        b = 20
        eta = 16.4
        d = 15

    elif bone == 'whole proximal tibia':
        sigma = 65
        L = 1200
        a = 35
        b = 25
        eta = 9.4
        d = 8.5

    elif bone == 'metaphysis proximal tibia':
        sigma = 70
        L = 1000
        a = 35
        b = 25
        eta = 15.5
        d = 12

    elif bone == 'epiphysis proximal tibia':
        sigma = 63
        L = 1300
        a = 35
        b = 15
        eta = 6.1
        d = 3.5

    elif bone == 'whole distal femur':
        sigma = 60
        d = 5
        a = 34
        b = 18
        eta = 5.4
        L = 1000

    elif bone == 'metaphysis distal femur':
        sigma = 60
        d = 5
        a = 30
        b = 30
        eta = 7.4
        L = 1000

    elif bone == 'epiphysis distal femur':
        sigma = 60
        d = 3.5
        a = 34
        b = 18
        eta = 3.8
        L = 1000

    else:
        raise ValueError("The parameters for the given bone are not yet implemented. Currently, the only options "
                         "available are: 'patella', 'proximal fibula', 'whole proximal tibia', 'epiphysis proximal tibia', "
                         "'metaphysis proximal tibia', 'whole distal femur', 'epiphysis distal femur' or "
                         "'metapysis distal femur'. Please choose one of those.")


    return sigma, L, a, b, eta, d


def model_concentric(mask_path, result_path, bone, degree, offset_q0, offset_r0, plot):
    """ Function creating 3D concentric trabecular structures. """

    # Parameters (depends on the specific region)
    white_noise_i_1 = 0  # Ni-1

    sigma, L, a, b, eta, d = parameters(bone)

    # List all files in the directory
    files = os.listdir(mask_path)

    # Filter out only BMP files
    bmp_files = [f for f in files if f.lower().endswith('.bmp')]

    np.random.seed(0)

    for mask_file in bmp_files:
        # Read in the trabecular bone mask slice per slice
        file_path = os.path.join(mask_path, mask_file)
        mask = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
        mask = rotate_clockwise(mask, degree=degree)
        mask = np.asarray(mask)

        map_size = mask.shape[0] # size of the images
        q0 = map_size // 2 - offset_q0
        r0 = map_size // 2 - offset_r0

        white_noise_i = generate_white_noise_map(map_size)

        # Connect the slices
        white_noise_i = white_noise_i + d * white_noise_i_1
        white_noise_i = white_noise_i / np.max(white_noise_i)

        # Create the slice
        point_wise_product = combine_gaussian_envelope_and_noise(
            map_size, sigma, white_noise_i)
        scaled_fourier_transform = fourier_transform(
            point_wise_product)
        R = R_concentric(map_size, a, b, q0, r0, L)
        result = resulting_function_T(R, scaled_fourier_transform, eta)

        # Mask the created structures
        image = cv2.bitwise_and(result, result, mask=mask)
        image = rotate_counterclockwise(image, degree=degree)

        # Binarize the images
        _, otsu_threshold = cv2.threshold(image, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = otsu_threshold

        # # plot
        if plot == "True":
            if plot == "True":
                middle_index = len(bmp_files) // 2
                middle_file = bmp_files[middle_index]
                middle_file_path = os.path.join(mask_path, middle_file)
                mask_middle = cv2.cvtColor(cv2.imread(middle_file_path), cv2.COLOR_BGR2GRAY)
                mask_middle = rotate_clockwise(mask_middle, degree=degree)
                mask_middle = np.asarray(mask_middle)
                image_middle = cv2.bitwise_and(result, result, mask=mask_middle)
                image_middle = rotate_counterclockwise(image_middle, degree=degree)
                _, otsu_threshold_middle = cv2.threshold(image_middle, 0, 255,
                                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                image_middle = otsu_threshold_middle
                plt.imshow(image_middle, cmap='gray')
                plt.show()

                os._exit(0)

        # Save the image
        output_path_file = os.path.join(result_path, mask_file)
        cv2.imwrite(output_path_file, image)

        # Save mask (for CTAn)
        mask_ouput_file = os.path.join(result_path, 'mask')
        os.makedirs(mask_ouput_file, exist_ok=True)

        mask_ouput_file = os.path.join(mask_ouput_file, mask_file)
        mask = rotate_counterclockwise(mask, degree=degree)
        cv2.imwrite(mask_ouput_file, mask)

        # set the i-1 as the noise map that was just used
        white_noise_i_1 = white_noise_i


def model_nonconcentric(mask_path, result_path, bone, degree, plot):
    """ Function creating 3D nonconcentric trabecular structures. """

    # Parameters (depends on the specific region)
    white_noise_i_1 = 0  # Ni-1

    sigma1, eta1, j01, k01, sigma2, eta2, j02, k02, d = parameters(bone)

    # List all files in the directory
    files = os.listdir(mask_path)

    # Filter out only BMP files
    bmp_files = [f for f in files if f.lower().endswith('.bmp')]

    np.random.seed(0)

    for mask_file in bmp_files:

        # Read in the trabecular bone mask slice per slice
        file_path = os.path.join(mask_path, mask_file)
        mask = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
        mask = rotate_clockwise(mask, degree=degree)
        mask = np.asarray(mask)

        map_size = mask.shape[0] # size of the images

        white_noise_i = generate_white_noise_map(map_size)

        # Connect the slices
        white_noise_i = white_noise_i + d * white_noise_i_1
        white_noise_i = white_noise_i / np.max(white_noise_i)

        # Create the slice -vertical
        point_wise_product = combine_lorentzian_envelope_and_noise(
            map_size, sigma1, white_noise_i, j01, k01)
        scaled_fourier_transform = fourier_transform(
            point_wise_product)
        R = R_nonconcentric(map_size)
        result1 = resulting_function_T(R, scaled_fourier_transform,
                                       eta1)

        # Create the slice -oblique
        point_wise_product = combine_lorentzian_envelope_and_noise(
            map_size, sigma2, white_noise_i, j02, k02)
        scaled_fourier_transform = fourier_transform(
            point_wise_product)
        R = R_nonconcentric(map_size)
        result2 = resulting_function_T(R, scaled_fourier_transform,
                                       eta2)

        # Combine the two structures
        result = cv2.add(result1, result2)

        # Mask the created structures
        image = cv2.bitwise_and(result, result, mask=mask)
        image = rotate_counterclockwise(image, degree=degree)

        # Binarize the images
        _, otsu_threshold = cv2.threshold(image, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = otsu_threshold

        # # plot
        if plot == "True":
            middle_index = len(bmp_files) // 2
            middle_file = bmp_files[middle_index]
            middle_file_path = os.path.join(mask_path, middle_file)
            mask_middle = cv2.cvtColor(cv2.imread(middle_file_path), cv2.COLOR_BGR2GRAY)
            mask_middle = rotate_clockwise(mask_middle, degree=degree)
            mask_middle = np.asarray(mask_middle)
            image_middle = cv2.bitwise_and(result, result, mask=mask_middle)
            image_middle = rotate_counterclockwise(image_middle, degree=degree)
            _, otsu_threshold_middle = cv2.threshold(image_middle, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_middle = otsu_threshold_middle
            plt.imshow(image_middle, cmap='gray')
            plt.show()

            os._exit(0)

            # Save the image
        output_path_file = os.path.join(result_path, mask_file)
        cv2.imwrite(output_path_file, image)

        # Save mask (for CTAn)
        mask_ouput_file = os.path.join(result_path, 'mask')
        os.makedirs(mask_ouput_file, exist_ok=True)

        mask_ouput_file = os.path.join(mask_ouput_file, mask_file)
        mask = rotate_counterclockwise(mask, degree=degree)
        cv2.imwrite(mask_ouput_file, mask)

        # set the i-1 as the noise map that was just used
        white_noise_i_1 = white_noise_i

if __name__ == "__main__":

    mask_path = sys.argv[1]
    result_path = sys.argv[2]
    os.makedirs(result_path, exist_ok=True)

    bone = sys.argv[3]

    if bone == 'patella': #or other non concentric bone added later
        if len(sys.argv) != 6:
            raise ValueError('Not the right amount of arguments were given. '
                          'As a reminder, the input should be '
                          'python model_trabecular_bone.py <"link to mask folder"> <"link to output folder"> <"bone"> '
                          '<degree> <"plot"> if you want to model a non-concentric bone.')
        else:
            degree = int(sys.argv[4])
            plot = sys.argv[5]

            model_nonconcentric(mask_path=mask_path, result_path=result_path, bone=bone, degree=degree, plot=plot)

    else:
        if len(sys.argv) != 8:
            raise ValueError('Not the right amount of arguments were given. '
                          'As a reminder, the input should be '
                          'python model_trabecular_bone.py <"link to mask folder"> <"link to output folder"> <"bone"> '
                          '<degree> <offset_q0> <offset_r0> <"plot"> if you want to model a concentric bone.')
        else:
            degree = int(sys.argv[4])
            offset_q0 = int(sys.argv[5])
            offset_r0 = int(sys.argv[6])
            plot = sys.argv[7]

            model_concentric(mask_path=mask_path, result_path=result_path, bone=bone, degree=degree, offset_q0=offset_q0,
                            offset_r0=offset_r0, plot=plot)
