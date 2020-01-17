"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from queue import Queue
from threading import Thread

import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions
from util import util


class ServerOptions(TestOptions):
    def initialize(self, parser):
        TestOptions.initialize(self, parser)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(dataset_mode='base')
        parser.set_defaults(label_nc=182)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(name='coco_pretrained')
        parser.add_argument('--port', type=int, default=8000, help='port')
        return parser


class QueueIterator:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        (input_data, is_raw) = input_queue.get()
        if is_raw:
            input_numpy = np.reshape(np.frombuffer(input_data, dtype=np.uint8, count=opt.crop_size * opt.crop_size),
                                     (opt.crop_size, opt.crop_size))
            label_tensor = transforms.ToTensor()(input_numpy) * 255.0
            original_size = (opt.crop_size, opt.crop_size)
            original_format = 'raw'
        else:
            original_image = Image.open(BytesIO(input_data))
            image = original_image.convert('L')
            if image.size != (opt.crop_size, opt.crop_size):
                image = image.resize((opt.crop_size, opt.crop_size), Image.NEAREST)
                image = image.crop((0, 0, opt.crop_size, opt.crop_size))
            label_tensor = transforms.ToTensor()(image) * 255.0
            original_size = original_image.size
            original_format = original_image.format
        label_tensor[label_tensor > opt.label_nc] = opt.label_nc
        return {
            'label': label_tensor,
            'instance': label_tensor,
            'image': label_tensor,
            'size': original_size,
            'format': original_format,
        }


# noinspection PyUnresolvedReferences
class CustomDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        super(CustomDataset).__init__()

    def __iter__(self):
        return QueueIterator()


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    # noinspection PyPep8Naming
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b'<form enctype="multipart/form-data" method="post">'
                         b'<input name="file" type="file"><input type="submit"></form>')

    # noinspection PyPep8Naming
    def do_POST(self):
        boundary = self.headers['content-type'].split("=")[1].encode()
        remaining_bytes = int(self.headers['content-length'])
        if remaining_bytes <= 0:
            self.send_response(400)
            return
        line = self.rfile.readline()
        remaining_bytes -= len(line)
        if not (boundary in line):
            self.send_response(400)
            return
        line = self.rfile.readline()
        remaining_bytes -= len(line)
        line = self.rfile.readline()
        remaining_bytes -= len(line)
        line = self.rfile.readline()
        remaining_bytes -= len(line)
        pre_line = self.rfile.readline()
        remaining_bytes -= len(pre_line)
        data = BytesIO()
        while remaining_bytes > 0:
            line = self.rfile.readline()
            remaining_bytes -= len(line)
            if boundary in line:
                pre_line = pre_line[0:-1]
                if pre_line.endswith(b'\r'):
                    pre_line = pre_line[0:-1]
                data.write(pre_line)
                data = data.getvalue()
                input_queue.put((data, bool(self.headers['x-raw'])))
                print('Added to input queue: ' + str(len(data)))
                data = output_queue.get()
                output_queue.task_done()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", len(data))
                self.end_headers()
                self.wfile.write(data)
                return
            else:
                data.write(pre_line)
                pre_line = line
        self.send_response(400)


opt = ServerOptions().parse()

model = Pix2PixModel(opt)
model.eval()

dataset = CustomDataset()
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=not opt.serial_batches,
    num_workers=int(opt.nThreads),
    drop_last=opt.isTrain
)


class WorkerThread(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        for _, data_i in enumerate(data_loader):
            image_numpy = util.tensor2im(model(data_i, mode='inference')[0])
            if len(image_numpy.shape) == 2:
                image_numpy = np.expand_dims(image_numpy, axis=2)
            if image_numpy.shape[2] == 1:
                image_numpy = np.repeat(image_numpy, 3, 2)
            if data_i['format'][0] == 'raw':
                image_numpy = np.append(np.zeros((image_numpy.shape[0], image_numpy.shape[1], 1), image_numpy.dtype),
                                        image_numpy, 2)
                data = image_numpy.tobytes()
            else:
                image = Image.fromarray(image_numpy)
                if image.size != data_i['size']:
                    image = image.resize(data_i['size'], Image.BICUBIC)
                data = BytesIO()
                image.save(data, format=data_i['format'][0])
                data = data.getvalue()
            print('Added to output queue: ' + str(len(data)))
            output_queue.put(data)
            input_queue.task_done()


input_queue = Queue()
output_queue = Queue()
worker_thread = WorkerThread()
worker_thread.setDaemon(True)
worker_thread.start()
httpd = HTTPServer(('0.0.0.0', opt.port), SimpleHTTPRequestHandler)
print('Server is ready')
httpd.serve_forever()
