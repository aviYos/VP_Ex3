import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# change IDs to your IDs.
ID1 = "315488171"
ID2 = "314756297"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# set random noise parameters
mu_x = 0.3
mu_y = 0.02
sigma_pos_x = 2.8
sigma_pos_y = 1.05
sigma_vel_x = 2.2
sigma_vel_y = 1.1

# number of histogram bits

NUMBER_OF_HISTOGRAM_BITS = 4

# Initial Settings
s_initial = [297,  # x center
             139,  # y center
             16,  # half width
             43,  # half height
             0,  # velocity x
             0]  # velocity y


def add_noise(state: np.ndarray) -> np.ndarray:
    """ Add normal random noise to our state
    """
    state[0, :] = state[0, :] + np.round(np.random.gamma(mu_x, sigma_pos_x, size=(1, N)))
    state[1, :] = state[1, :] + np.round(np.random.gamma(mu_y, sigma_pos_y, size=(1, N)))
    state[4, :] = state[4, :] + np.round(np.random.gamma(mu_x, sigma_vel_x, size=(1, N)))
    state[5, :] = state[5, :] + np.round(np.random.gamma(mu_y, sigma_vel_y, size=(1, N)))
    return state


def propagate_state(state: np.ndarray) -> np.ndarray:
    """ propagate our state with time.
    """
    # pos_x + vel_x
    state[0, :] = state[0, :] + state[4, :]
    # pos_y + vel_y
    state[1, :] = state[1, :] + state[5, :]
    return state


def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    s_prior = s_prior.astype(float)
    propagated_state = propagate_state(s_prior)
    state_drifted = add_noise(propagated_state).astype(int)
    return state_drifted


def slice_image(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """ slice_image by its center and widths
    """
    x_center, y_center, half_width, half_height, _, _ = state
    return image[y_center - half_height: y_center + half_height, x_center - half_width: x_center + half_width]


def calc_histogram(image: np.ndarray, number_of_bins: int) -> np.ndarray:
    histogram = np.zeros((number_of_bins, number_of_bins, number_of_bins))
    red_channel, green_channel, blue_channel = cv2.split(image)
    h, w = red_channel.shape

    for height in range(h):
        for width in range(w):
            histogram[red_channel[height, width], green_channel[height, width], blue_channel[height, width]] += 1

    return histogram


def create_quantizied_histogram(image: np.ndarray) -> np.ndarray:
    number_of_bins = int(np.power(2, NUMBER_OF_HISTOGRAM_BITS))
    quantized_image = np.uint8(np.floor(np.divide(image, number_of_bins)))
    histogram = calc_histogram(quantized_image, number_of_bins)
    return histogram


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    state = np.floor(state)
    state = state.astype(int)
    sliced_image = slice_image(image, state)
    hist = create_quantizied_histogram(sliced_image)
    hist = np.reshape(hist, 16 * 16 * 16)

    # normalize
    hist = hist / sum(hist)

    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """

    # create initial state
    s_init = np.array([297, 139, 16, 43, 0, 0])

    S_next = np.zeros((6, N))
    for particle_idx in range(N):
        r = np.random.uniform(0, 1)
        j = np.argmax(cdf >= r)
        S_next[:, particle_idx] = previous_state[:, j]
    return S_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """

    # BC = np.sum(np.sqrt(p * q))
    # distance = np.float(-np.log(BC))

    distance = np.exp(20 * np.sum(np.sqrt(p * q)))

    return distance


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                   frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                   ) -> tuple:
    fig, ax = plt.subplots(1)
    image = image[:, :, ::-1]
    plt.imshow(image)
    plt.title(ID + " - Frame mumber = " + str(frame_index))

    # Avg particle box
    avg_W = np.average(W)
    avg_index = (np.abs(W-avg_W)).argmin()
    (x_avg, y_avg, w_avg, h_avg) = state[:4, avg_index]
    x_avg -= w_avg
    y_avg -= h_avg
    w_avg *= 2
    h_avg *= 2

    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    max_index = np.argmax(W)
    (x_max, y_max, w_max, h_max) = state[:4, max_index]
    x_max -= w_max
    y_max -= h_max
    w_max *= 2
    h_max *= 2

    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show(block=False)

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state


def compute_weights(q: np.ndarray, image: np.ndarray, S: np.ndarray) -> np.ndarray:
    W = np.zeros((1, N))
    for particle_idx in range(N):
        current_p = compute_normalized_histogram(image, S[:, particle_idx])
        W[0, particle_idx] = bhattacharyya_distance(current_p, q)
    W /= np.sum(W)
    return W


def compute_CDF(W: np.ndarray) -> np.ndarray:
    C = np.zeros((1, N))
    C[0, 0] = W[0, 0]
    for particle_idx in range(1, N - 1):
        C[0, particle_idx] = W[0, 0:particle_idx + 1].sum()
    C[0, N - 1] = 1
    return C


def calculate_weights_and_CDF(q: np.ndarray, image: np.ndarray, S: np.ndarray) -> (np.ndarray, np.ndarray):
    W = compute_weights(q, image, S)
    C = compute_CDF(W)
    return W, C


def main():
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    # YOU NEED TO FILL THIS PART WITH CODE:
    W, C = calculate_weights_and_CDF(q, image, S)

    # COMPUTE S_next, the next filter state
    s_next = sample_particles(S, C)

    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    for image_name in image_name_list[1:]:

        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # YOU NEED TO FILL THIS PART WITH CODE:
        W, C = calculate_weights_and_CDF(q, image, S)

        # CREATE DETECTOR PLOTS
        images_processed += 1
        if 0 == images_processed % 10:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
