import cv2
import threading
import queue
import time
from .interfaces import BaseModule
from .logger import ResearchLogger

class ThreadedVideoInterface(BaseModule):
    def __init__(self, source=0, queue_size=128):
        super().__init__("VideoInterface")
        self.source = source
        self.queue_size = queue_size
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = None
        self.cap = None
        self.logger = ResearchLogger()

    def initialize(self) -> bool:
        self.logger.log("INFO", f"Initializing Video Capture from source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.logger.log("CRITICAL", "Failed to open video source.")
            return False
        
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        return True

    def _update(self):
        while not self.stopped:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    self.logger.log("WARNING", "Video stream ended or disconnected.")
                    break
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01) 

    def process(self, _=None):
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None

    def shutdown(self):
        self.stopped = True
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        self.logger.log("INFO", "Video Interface shut down successfully.")
