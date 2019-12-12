from skimage import io

def psnr(original_image, output_image):
	original = measure.block_reduce(io.imread(original_image), [4,4])
	output = io.imread(output_image)
	return metrics.peak_signal_noise_ratio(original, output)

