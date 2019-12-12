import glob
from skimage import io
from skimage import metrics
from skimage.transform import resize 


def calculate_psnr(original_image, output_image):
	'''
    Parameters:
    original_image :: String - file path to original image
    output_image :: String - file path to output image

    Returns:
    psnr :: Float - peak signal to noise ratio for the two images
    '''
	output = io.imread(output_image)
	output_scaled = output / 256

	original = io.imread(original_image)
	original_resized = resize(original, (output_scaled.shape[0], output_scaled.shape[1]), anti_aliasing = True)
	
	return metrics.peak_signal_noise_ratio(original_resized, output_scaled)

def average_psnr(input_dir, output_dir):
	'''
    Parameters:
    input_dir :: String - file path to directory of input images
    output_dir :: String - file path to directory of output images

	This works on images with extensions jpg, JPG, jpeg, and JPEG. Input and 
	output images are paired up based on lexographical order in their directories

    Returns:
    psnr :: Float - peak signal to noise ratio for the two images
    '''
	input_files = sorted(glob.glob(input_dir + "/*.jpg") + glob.glob(input_dir + "/*.JPG")
		+ glob.glob(input_dir + "/*.jpeg") + glob.glob(input_dir + "/*.JPEG"))
	output_files = sorted(glob.glob(output_dir + "/*.jpg") + glob.glob(output_dir + "/*.JPG")
		+ glob.glob(output_dir + "/*.jpeg") + glob.glob(output_dir + "/*.JPEG"))

	assert len(input_files) == len(output_files), "different number of output and input files"
	print("found " , len(input_files), " input/output file pairs")

	psnr_values = []
	for i in range(len(input_files)):
		psnr = calculate_psnr(input_files[i], output_files[i])
		psnr_values.append(psnr)

	return sum(psnr_values) / len(psnr_values)

def main():
	print("average psnr: ", average_psnr("./psnr_inputs", "./psnr_outputs"))

if __name__ == "__main__":
	main()






