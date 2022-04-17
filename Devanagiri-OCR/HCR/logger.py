import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf

class Logger:
    def __init__(self):
        pass

    @staticmethod
    def init():
        print("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        print(f"Devanagiri OCR")
        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")
        Logger.getGPUStatus()

    @staticmethod
    def menu():
        print("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")
        print("1. Train Devanagiri OCR")
        print("2. Test Devanagiri OCR")
        print("Q/q To exit\n")
        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        
        inp_option = input("Your selection: ")
        return inp_option
    
    @staticmethod
    def getGPUStatus():
        #CPU + GPU
        list_of_devices = tf.config.get_visible_devices()

        #CUDA GPUs only
        is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)

        if is_cuda_gpu_available:
            Logger.success("Running in GPU mode")
            Logger.info("Device name: {}".format(tf.test.gpu_device_name()))

        else:
            Logger.info(f"TF Devices: {list_of_devices}")
            Logger.warning("Running in CPU mode")

    @staticmethod
    def success(msg):
        msg = "✅ " + msg
        print(msg)
    
    @staticmethod
    def info(msg):
        msg = "ℹ️ " + msg
        print(msg)
    
    @staticmethod
    def warning(msg):
        msg = "⚠️  " + msg
        print(msg)
    
    @staticmethod
    def error(msg):
        msg = "🛑 " + msg
        print(msg)