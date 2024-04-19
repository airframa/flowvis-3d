from PySide6.QtCore import QThread
import time

class Worker(QThread):
    def run(self):
        print("Thread started")
        time.sleep(2)
        print("Thread running")

if __name__ == "__main__":
    thread = Worker()
    thread.start()
    time.sleep(1)
    if thread.isRunning():
        print("Thread is still running")
    thread.wait()
    print("Thread has finished")
