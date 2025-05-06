import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import utils_scratch
from collections import namedtuple
import os
import cv2
import math
import config
from utils_scratch import centerpoint, memory_check
from trajectory_lstm import predict_next_bounding_box
 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(

            nn.Conv2d(1, 96, kernel_size=11,stride=1),
            nn.BatchNorm2d(96),
            #nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

        )

    def forward(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        return output

'''
class MODMLSTM(nn.Module):

    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers):
        super(MODMLSTM, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Index the last time step
        out = out[:, -1, :]
        return out
'''

class MODM():




    def __init__(self,
                 lstm1_input_size=512,
                 lstm2_input_size = 512,
                 lstm1_hidden_size=512,
                 lstm2_hidden_size = 512):

        super().__init__()

        self.training = True
        self.feature_extraction = CNN()
        # Define the LSTM layer

        self.LSTM1= nn.LSTM(lstm1_input_size , lstm1_hidden_size , batch_first=True)



    def forward(self,source, prev_state,prev_box):

        input = self.feature_extraction(source)
        memory = prev_state.memory
        #Attention
        input_shape = input.size()
        query = prev_state.h_t
        input_transform = nn.Conv2d(input.shape[1], input.shape[1], kernel_size=1, stride=1, bias=False)
        dense_layer = nn.Linear(in_features=input_shape[-1], out_features=input_shape[-1])
        query_transform = dense_layer(query)
        query_transform = torch.unsqueeze(torch.unsqueeze(query_transform, 1),1)
        addition = torch.tanh(input_transform + query_transform)
        conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        addition_transform = conv_layer(addition)
        addition_shape = addition_transform.size()
        score = F.softmax(addition_transform.view(addition_shape[0], -1), dim=1)
        lstm_input = torch.sum(input * score, [2, 3])


        #LSTM
        h0 = torch.zeros(num_layers=1, input_shape=512, lstm1_hidden_size=512)
        # Initialize cell state
        c0 = torch.zeros(num_layers=1, input_shape=512, lstm1_hidden_size=512)
        lstm_output, lstm_state = self.LSTM1(lstm_input,(self.h0,self.c0))

        h_t = lstm_state[0]

        read_key, read_strength, control_factors, write_decay, residual_vector = utils_scratch._transform_input(lstm_output)

        # Read from memory

        read_weight = self._read_weights(read_key, read_strength, prev_state.memory)
        read_weight_expand = read_weight.view(-1, self._memory_size, 1, 1, 1)
        final_template = torch.sum(read_weight_expand * memory, dim=[1])

        # Calculate the allocation weight
        allocation_weight = utils_scratch.calc_allocation_weight(prev_state.usage, self._memory_size)

        # Calculate the write weight for the next frame writing
        write_weight = self._write_weights(control_factors, read_weight, allocation_weight)

        # Update usage using read & write weights and previous usage
        usage = utils_scratch.update_usage(write_weight, read_weight, prev_state.usage)

        memory = utils_scratch._reset_and_write(prev_state.memory, prev_state.write_weight,
                                  prev_state.write_decay, prev_state.control_factors, memory)


        State = namedtuple('Cur_State', [
            'memory',
            'read_weight',
            'write_weight',
            'control_factors',
            'write_decay',
            'usage',
            'h_t'
        ])
        Cur_state = State(
            memory=memory,
            write_weight=write_weight,
            read_weight=read_weight,
            control_factors=control_factors,
            write_decay=write_decay,
            usage=usage,
            h_t = h_t)

        bbox = np.array(prev_box)
        target_pos = bbox[0:2] + bbox[2:4] / 2
        target_size = bbox[2:4]
        tarcoor, tarsize, scale_adj = utils_scratch.findbox(final_template,target_size,target_pos)
        bbox = np.hstack([tarcoor - tarsize / 2, scale_adj])

        return Cur_state,bbox


def load_seq_config(seq_name, trackidx, CellNum):
    GTFile = 'groundtruth_rect' + str(CellNum) + '.txt'

    src = os.path.join(config.otb_data_dir + config.code_seq, GTFile)

    # src = os.path.join(config.otb_data_dir,seq_name,'groundtruth_rect1.txt')
    gt_file = open(src)
    lines = gt_file.readlines()
    gt_rects = []
    for gt_rect in lines:
        rect = [int(v) for v in gt_rect[:-1].split(',')]
        gt_rects.append(rect)

    init_rect = gt_rects[trackidx]
    img_path = os.path.join(config.otb_data_dir + config.image_seq)
    img_names = sorted(os.listdir(img_path))
    s_frames = [os.path.join(img_path, img_name) for img_name in img_names]

    return init_rect, s_frames


def display_result(image, pred_boxes, frame_idx, cp, seq_name=None):
    if len(image.shape) == 3:
        r, g, b = cv2.split(image)
        image = cv2.merge([b, g, r])
    pred_boxes = pred_boxes.astype(int)
    points = np.array(cp[1:frame_idx])
    cv2.rectangle(image, tuple(pred_boxes[0:2]), tuple(pred_boxes[0:2] + pred_boxes[2:4]), (0, 0, 255), 2)

    cv2.putText(image, 'Frame: %d' % frame_idx, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255))
    # cv2.polylines(image, np.int32([points]), 0, (255,255,255))
    cv2.imshow('tracker', image)



def clear():
    for key, value in globals().items():

        if callable(value) or value.__class__.__name__ == "module":
            continue

        del globals()[key]


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)

def run_tracker(lower,upper):


    tracker = MODM(lstm1_input_size=512,lstm1_hidden_size=512)
    SPCHANGE = []
    timeflag = 0
    for cn in range(lower, upper + 1):
        #######################################################
        trackidx = 0
        #######################################################

        init_rect, s_frames = load_seq_config('', trackidx, cn)

        ResFile = 'CTRes' + str(cn) + '.txt'

        file_save = open(os.path.join(config.otb_data_dir + config.code_seq, ResFile), mode='w')

        # response_save=open('/home/ran/Trails/N2DH-GOWH1/02/RESFILE/resultresponse24.txt', mode= 'w')
        fn = os.path.join(config.otb_data_dir + config.code_seq, ResFile)
        # fnp24='/home/ran/Trails/N2DH-GOWH1/02/RESFILE/resultresponse6.txt'
        linenp = np.loadtxt(fn)
        cp = centerpoint(linenp)
        cplist = []
        cplist.append(cp)
        rediuslist = []

        bbox = init_rect
        init_center = [math.floor(bbox[0] + 0.5 * bbox[2]), math.floor(bbox[1] + 0.5 * bbox[3])]

        res = []
        res.append(bbox)


        response_dir = os.path.join(config.otb_data_dir + config.code_seq, 'resultresponse/v1/' + str(cn))
        mkdir(response_dir)

        templatenorm = []

        for idx in range(trackidx, len(s_frames)):

            tracker.idx = idx

            if idx < 2:
                redius = 0
                center = init_center
            else:
                center = [cplist[idx][0], cplist[idx][1]]
                P1_X = cplist[idx][0]
                P1_Y = cplist[idx][1]
                P2_X = cplist[idx - 1][0]
                P2_Y = cplist[idx - 1][1]

                redius = math.sqrt((P1_X - P2_X) * (P1_X - P2_X) + (P1_Y - P2_Y) * (P1_Y - P2_Y))
                rediuslist.append(redius)

                # print(np.mean(rediuslist))
                # print('/')
                # print(redius-np.mean(rediuslist))

                SPCHANGE.append(redius)

            # bbox, cur_frame,response,instance= tracker.track(s_frames[idx])
            bbox, targetpos, cur_frame, response, instance, memory, templatenew = MODM(s_frames[idx], redius, center, idx)

            bbox = predict_next_bounding_box(res[:-1], model_path="trajectory_model.pth", device="cpu")

            ############ Motion constraint######


            bbox = bbox.astype(np.int)
            res.append(bbox.tolist())
            cp0 = [math.floor(bbox[0] + bbox[2] * 0.5), math.floor(bbox[1] + bbox[3] * 0.5)]
            cp = targetpos
            cplist.append(cp)

            # print(" ".join(str(i) for i in bbox))
            bboxoutput = " ".join(str(i) for i in bbox)

            print('frame ', idx, ':')
            # print(bboxoutput)
            # print(cplist[idx])
            if idx > 0:
                print(res[idx])
            # print(response)
            response_shape = config.response_up * config.response_size
            response_map = response.reshape(response_shape, response_shape)  # (272,272)
            # exec("np.savetxt(os.path.join(config.otb_data_dir,'tracking/resultresponse"+str(idx)+".txt'),response_map)")
            instance_map = instance.reshape(255, 255, 3)

            # print(np.shape(response))
            response_show = 255 / np.max(response_map) * response_map
            instance_show = 255 / np.max(instance_map) * instance_map
            # print(np.max(response_map))

            RESPFile = str(idx) + '.png'

            response_save = os.path.join(response_dir, RESPFile)

            exec("cv2.imwrite(response_save,response_show)")
            # exec("cv2.imwrite(os.path.join(config.otb_data_dir,'tracking/resultinstance"+str(idx)+".jpg'),instance_show)")
            # response_save.write('\n')

            # file_save.write(np.array2string(bbox))
            file_save.write(bboxoutput)  # bboxoutput-output as 12 13 13 13 str(bbox)-output as [12 13 13 13]
            file_save.write('\n')
            # bbox1=linenp
            display_result(cur_frame, bbox, idx, cp)  ###display results
            timeflag += 1


            if idx >= 1:
                # plt.imshow(response_map)
                # plt.show()
                # cv2.imwrite(response_map,'/home/ran/Desktop/1.png')
                a = 1

            if idx >= 8:
                memory_check(memory, idx, trackidx)
                x_norm = np.linalg.norm(templatenew)
                # print(x_norm)

                templatenorm.append(x_norm)

        type = 'rect'

        file_save.close()


    return SPCHANGE, res, type, templatenorm


def speedanalysis(cn, SP):
    Speedtotal = []
    Speedchangetotal = []
    for cidx in range(0, cn):
        SPCHANGE = []
        SPEEDORI = []
        for i in range(0, int(len(SP) / cn)):

            fidx = int(i + cidx * len(SP) / cn)

            if i < 1:
                spchange = 0

            else:
                spchange = SP[fidx] - SP[fidx - 1]

            speedori = SP[fidx]

            # speed and change in current cell
            SPEEDORI.append(speedori)
            SPCHANGE.append(spchange)

            # speed and change among all cells
            Speedtotal.append(speedori)
            Speedchangetotal.append(spchange)

        exec("Speedfile='Speed" + str(cidx + 1) + ".txt'")
        speed_save = open(os.path.join(config.otb_data_dir + config.code_seq, Speedfile), mode='w')
        for speeds in SPEEDORI:
            speeds = str(speeds)
            speed_save.write(speeds)  # bboxoutput-output as 12 13 13 13 str(bbox)-output as [12 13 13 13]
            speed_save.write('\n')

        exec("Speedchfile='Speedchange" + str(cidx + 1) + ".txt'")
        speedch_save = open(os.path.join(config.otb_data_dir + config.code_seq, Speedchfile), mode='w')
        for speedchs in SPCHANGE:  # SPCHSANGE
            speedchs = str(speedchs)
            speedch_save.write(speedchs)  # bboxoutput-output as 12 13 13 13 str(bbox)-output as [12 13 13 13]
            speedch_save.write('\n')

    return Speedtotal, Speedchangetotal


if __name__ == '__main__':

    cell_number = 1  #################
    cn = cell_number
    res = []
    SP, res, type, fps, templatenorm = run_tracker(1, cell_number)
    print(templatenorm)

    print(fps)

    # sp,spc = speedanalysis(cell_number,SP)

    # print(SPCHANGE)
    Speedtotal = []
    Speedchangetotal = []
    for cidx in range(0, cn):
        SPCHANGE = []
        SPEEDORI = []
        for i in range(0, int(len(SP) / cn)):

            fidx = int(i + cidx * len(SP) / cn)

            if i < 1:
                spchange = 0

            else:
                spchange = SP[fidx] - SP[fidx - 1]

            speedori = SP[fidx]

            # speed and change in current cell
            SPEEDORI.append(speedori)
            SPCHANGE.append(spchange)

            # speed and change among all cells
            Speedtotal.append(speedori)
            Speedchangetotal.append(spchange)

        exec("Speedfile='Speed" + str(cidx + 1) + ".txt'")
        speed_save = open(os.path.join(config.otb_data_dir + config.code_seq, Speedfile), mode='w')
        for speeds in SPEEDORI:
            speeds = str(speeds)
            speed_save.write(speeds)  # bboxoutput-output as 12 13 13 13 str(bbox)-output as [12 13 13 13]
            speed_save.write('\n')

        exec("Speedchfile='Speedchange" + str(cidx + 1) + ".txt'")
        speedch_save = open(os.path.join(config.otb_data_dir + config.code_seq, Speedchfile), mode='w')
        for speedchs in SPCHANGE:  # SPCHSANGE
            speedchs = str(speedchs)
            speedch_save.write(speedchs)  # bboxoutput-output as 12 13 13 13 str(bbox)-output as [12 13 13 13]
            speedch_save.write('\n')

    # DrawCTCRES(cell_number)
