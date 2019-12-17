# -*- coding: utf-8 -*-

import tensorflow as tf
from read_utils import TextConverter
from model import CharRNN
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_path', '', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', '', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 30, 'max length to generate')

poemList = []
mutex = threading.Lock()

class Resquest(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json;charset=utf-8')
        self.end_headers()
        if 0 < len(poemList):
            try:
                mutex.acquire()
                poem = poemList[0]
                poemList.pop(0)
                #print(poem)
                self.wfile.write(str(poem).encode(encoding='utf-8'))
            finally:
                mutex.release()
        else:
            #print("nothing")
            self.wfile.write(str("nothing to say").encode(encoding='utf-8'))


class fillPoem(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID

    def run(self):
        converter = TextConverter(filename=FLAGS.converter_path)
        if os.path.isdir(FLAGS.checkpoint_path):
            FLAGS.checkpoint_path = \
                tf.train.latest_checkpoint(FLAGS.checkpoint_path)

        model = CharRNN(converter.vocab_size, sampling=True,
                        lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                        use_embedding=FLAGS.use_embedding,
                        embedding_size=FLAGS.embedding_size)
        model.load(FLAGS.checkpoint_path)
        start = converter.text_to_arr(FLAGS.start_string)
        while True:
            if 5 > len(poemList):
                try:
                    mutex.acquire()
                    arr = model.sample(FLAGS.max_length, start, converter.vocab_size)
                    poemList.append(converter.arr_to_text(arr))
                finally:
                    mutex.release()
                    time.sleep(1)

def main(_):
    thread = fillPoem(1)
    thread.start()

    #FLAGS.start_string = FLAGS.start_string.decode('utf-8')
    # converter = TextConverter(filename=FLAGS.converter_path)
    # if os.path.isdir(FLAGS.checkpoint_path):
    #     FLAGS.checkpoint_path =\
    #         tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    #
    # model = CharRNN(converter.vocab_size, sampling=True,
    #                 lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
    #                 use_embedding=FLAGS.use_embedding,
    #                 embedding_size=FLAGS.embedding_size)
    # model.load(FLAGS.checkpoint_path)
    # start = converter.text_to_arr(FLAGS.start_string)
    # arr = model.sample(FLAGS.max_length, start, converter.vocab_size)
    # print(converter.arr_to_text(arr))

    host = ('0.0.0.0', 22222)
    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()



if __name__ == '__main__':
    tf.app.run()
