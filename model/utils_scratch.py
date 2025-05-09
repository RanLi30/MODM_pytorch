import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import config
 

class LoadableModel(nn.Module):
    """
    Base class for easy pytorch model loading without having to manually
    specify the architecture configuration at load time.

    We can cache the arguments used to the construct the initial network, so that
    we can construct the exact same network when loading from file. The arguments
    provided to __init__ are automatically saved into the object (in self.config)
    if the __init__ method is decorated with the @store_config_args utility.
    """

    # this constructor just functions as a check to make sure that every
    # LoadableModel subclass has provided an internal config parameter
    # either manually or via store_config_args
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'config'):
            raise RuntimeError('models that inherit from LoadableModel must decorate the '
                               'constructor with @store_config_args')
        super().__init__(*args, **kwargs)

    def save(self, path):
        """
        Saves the model configuration and weights to a pytorch file.
        """
        # don't save the transformer_grid buffers - see SpatialTransformer doc for more info
        sd = self.state_dict().copy()
        grid_buffers = [key for key in sd.keys() if key.endswith('.grid')]
        for key in grid_buffers:
            sd.pop(key)
        torch.save({'config': self.config, 'model_state': sd}, path)

    @classmethod
    def load(cls, path, device):
        """
        Load a python model configuration and weights.
        """
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        return model


def get_key_feature(input, is_train, name, slot_size):
    input_shape = list(input.size())

    if len(input_shape) > 4:
        input = input.view(-1, *input_shape[2:])

    controller_input = F.avg_pool2d(input, kernel_size=slot_size[:2], stride=(1, 1), padding=0)

    if len(input_shape) > 4:
        c_shape = list(controller_input.size())
        controller_input = controller_input.view(input_shape[:2] + c_shape[1:])

    return controller_input





def _reset_and_write(memory, write_weight, write_decay, control_factors, values):
    weight_shape = write_weight.size()
    write_weight = write_weight.view(*weight_shape, 1, 1, 1)
    decay = write_decay * control_factors[:, 1].unsqueeze(1) + control_factors[:, 2].unsqueeze(1)
    decay_expand = decay.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    decay_weight = write_weight * decay_expand

    memory *= 1 - decay_weight
    values = values.unsqueeze(1)
    memory += decay_weight * values

    return memory
def cosine_similarity(memory, keys, strengths, strength_op=F.softplus):
    # Calculates the inner product between the query vector and words in memory.
    keys = keys.unsqueeze(1)
    dot = torch.matmul(keys, memory.permute(0, 2, 1))

    # Outer product to compute the denominator (euclidean norm of query and memory).
    memory_norms = _vector_norms(memory)
    key_norms = _vector_norms(keys)
    norm = torch.matmul(key_norms, memory_norms.unsqueeze(1))

    # Calculates cosine similarity between the query vector and words in memory.
    similarity = dot / (norm + 1e-6)

    return _weighted_softmax(similarity.squeeze(1), strengths, strength_op)

# Define the _vector_norms function used for memory and keys.
def _vector_norms(tensor):
    return torch.norm(tensor, dim=-1)

# Define the _weighted_softmax function that you are using.
def _weighted_softmax(similarity, strengths, strength_op):
    weighted_similarity = similarity * strength_op(strengths)
    return F.softmax(weighted_similarity, dim=-1)

def _transform_input(self, input):
    control_factors = torch.nn.functional.softmax(nn.Linear(input.size(1), 3)(input), dim=-1)
    write_decay = torch.sigmoid(nn.Linear(input.size(1), 1)(input))
    residual_vector = torch.sigmoid(nn.Linear(input.size(1), self._slot_size[2])(input))

    read_key = nn.Linear(input.size(1), 256)(input)

    read_strength = nn.Linear(input.size(1), 1)(input)
    read_strength = torch.ones(read_strength.size())  # Assuming bias_initializer=tf.ones_initializer()

    return read_key, read_strength, control_factors, write_decay, residual_vector

def _write_weights(self, control_factors, read_weight, allocation_weight):
    return control_factors[:, 1].unsqueeze(1) * read_weight + control_factors[:, 2].unsqueeze(1) * allocation_weight


def get_key_feature(input, is_train, name):
    input_shape = input.size()

    if len(input_shape) > 4:
        input = input.view(-1, *input_shape[2:])

    controller_input = nn.AvgPool2d(input_shape[2:4], stride=(1, 1))(input)

    if len(input_shape) > 4:
        c_shape = controller_input.size()
        controller_input = controller_input.view(input_shape[0:2] + c_shape[1:])

    return controller_input

def _read_weights(self, read_key, read_strength, memory):
    memory_key = get_key_feature(memory, self._is_train, 'memory_key').squeeze(dim=(2, 3))

    return cosine_similarity(memory_key, read_key, read_strength)

def update_usage(write_weights, read_weights, prev_usage):
    usage = 0.99 * prev_usage + write_weights + read_weights
    return usage

##
def calc_allocation_weight(usage, memory_size):
    nonusage = 1 - usage
    sorted_nonusage, indices = torch.topk(nonusage, k=1, dim=1)
    allocation_weights = torch.zeros(usage.size())
    allocation_weights.scatter_(1, indices, memory_size)
    return allocation_weights

##
def findbox(responses, target_size, target_pos):
    scale_steps = list(range(math.ceil(1 / 2) - 1, math.floor(1 / 2) + 1))
    scales = np.power(1.1, scale_steps) #scale_multiplayer

    up_response_size = 17 * 1
    window = np.matmul(np.expand_dims(np.hanning(up_response_size), 1),np.expand_dims(np.hanning(up_response_size), 0)).astype(np.float32)


    current_scale_idx = math.floor(1/2)
    best_scale_idx = current_scale_idx
    best_peak = -math.inf

    extend_size = target_size + 0.5 * (target_size[0] + target_size[1])
    z_size = np.sqrt(np.prod(extend_size))
    z_size = np.repeat(z_size, 2, 0)

    z_scale = 127 / z_size
    delta_size = 255 - 127
    x_size = delta_size / z_scale + z_size

    for s_idx in range(1):
        this_response = responses[s_idx].copy()

        # penalize the change of scale
        if s_idx != current_scale_idx:
            this_response *= 0.97  #scale penalty
        this_peak = np.max(this_response)
        if this_peak > best_peak:
            best_peak = this_peak
            best_scale_idx = s_idx
    response = responses[best_scale_idx]

    x_roi_size_orig = x_size[best_scale_idx]

    # make response sum to 1
    response -= np.min(response)
    response /= np.sum(response)


    # apply window
    response = (1 - 0.15) * response + 0.15 * window

    max_idx = np.argsort(response.flatten())
    max_idx = max_idx[-1:]

    x = max_idx % up_response_size
    y = max_idx // up_response_size
    position = np.vstack([x, y]).transpose()

    shift_center = position - up_response_size / 2
    shift_center_instance = shift_center * 8 / 1 #stride / response_up
    shift_center_orig = shift_center_instance * np.expand_dims(x_roi_size_orig, 0) / 255
    target_pos = np.mean(target_pos + shift_center_orig, 0)

    target_size_new = target_size * scales[best_scale_idx]
    target_size = (1 - 0.6) * target_size + 0.6 * target_size_new

    return target_pos, target_size, best_scale_idx


def centroid(i, bbox):
    # bbox=bbox.astype(int)
    # centroid=np.array((bbox[i,0]+bbox[i,2]/2,bbox[i,1]+bbox[i,3]/2))
    centroid_x = math.floor(bbox[i, 0] + bbox[i, 2] / 2)
    centroid_y = math.floor(bbox[i, 1] + bbox[i, 3] / 2)
    return centroid_x, centroid_y


def centerpoint(linenp1):
    center = []
    t = []
    a = []
    for idx in range(len(linenp1)):
        #        t=centroid(idx,linenp1)
        x, y = centroid(idx, linenp1)
        x = math.floor(x)
        y = math.floor(y)
        t = [x, y]
        # t=t.astype(int)
        center.append(t)

        a = np.array(center)
        a = a.astype(int)
        print(a)
        # file_handle.write(np.array2string(a))
        # file_handle.write('\n')
        # print(a)
    #      t.append(t)
    # tbbox=tuple(linenp1[idx])
    return a

def calc_z_size(target_size):
    # calculate roi region
    if config.fix_aspect:
        extend_size = target_size + config.context_amount * (target_size[0] + target_size[1])
        z_size = np.sqrt(np.prod(extend_size))
        z_size = np.repeat(z_size, 2, 0)
    else:
        z_size = target_size * config.z_scale

    return z_size

def calc_x_size(z_roi_size):
    # calculate roi region
    z_scale = config.z_exemplar_size / z_roi_size
    delta_size = config.x_instance_size - config.z_exemplar_size
    x_size = delta_size / z_scale + z_roi_size

    return x_size


def memory_check(memoryori, idx, cellnumber):
    memory_number = len(memoryori)

    for j in range(0, memory_number):
        mem = memoryori
        # _range=np.max(memoryori[j])-np.min(memoryori[j])
        # mem=(memoryori[j]-np.min(memoryori[j]))/_range*256
        # mem.astype(int)
        memoryout = []
        for c in range(0, len(mem[0][0])):
            temp = mem[:, :, c].flatten(order='F')
            memoryout.append(temp)
        out = np.array(memoryout)

        exec("cv2.imwrite('/home/ran/Pictures/memory/cell" + str(cellnumber) + "_" + str(idx) + "_" + str(
            j) + ".tiff',out)")
