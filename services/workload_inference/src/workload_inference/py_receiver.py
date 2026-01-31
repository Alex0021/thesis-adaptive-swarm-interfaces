from workload_inference.utilities import ConsoleManager
import threading
import mmap
from typing import Any
import workload_inference.data_structures as dts
import time
import numpy as np
import logging
import zmq
from typing import Callable

METADATA_BLOCK_NAME = "TobiiUnityMetadata2"
GAZE_DATA_BLOCK_NAME = "TobiiUnityGazeData2"
GAZE_DATA_BLOCK_CNT = 100

type Listener = Callable[[list[dts.GazeData]], None]

class PyReceiverBase:
    """
    Base class for Gaze Data Receivers.
    """

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._lock: threading.Lock = threading.Lock()
        self._running: bool = False
        self._ready: bool = False

        self._listeners: list[Listener] = []
        """Listeners for gaze data updates. Each listener is a callable function that takes a list of GazeData instances as an argument."""
        
        self._monitor: Monitor = Monitor()
        self._console: ConsoleManager = ConsoleManager()

        self._logger = logging.getLogger("PyReceiverBase")
    
    def start(self) -> None:
        raise NotImplementedError()

    def stop(self) -> None:
        raise NotImplementedError()

    def register_listener(self, listener: Callable[[list[dts.GazeData]], None]) -> None:
        """
        Register a listener to receive gaze data updates.

        Args:
            listener (Callable[[list[dts.GazeData]], None]): A callable to receive a list of GazeData instances.
        """
        with self._lock:
            self._listeners.append(listener)
        
    def clear_listeners(self) -> None:
        """
        Clear all registered listeners.
        """
        with self._lock:
            self._listeners.clear()

    def pretty_print_gaze_data(self, gaze_data: dts.GazeData) -> None:
        """
        Pretty print the gaze data.
        """
        print('\r--------------' + ' '*20)
        for key, value in gaze_data.__dict__.items():
            print(f"  {key}: {value}")


class SMReceiver(PyReceiverBase):
    """
    Shared Memory Receiver for Gaze Data.
    """

    def __init__(self):
        super().__init__()
        self._metadata_block: mmap.mmap | None = None
        self._gaze_data_block: mmap.mmap | None = None
        self._gaze_data_ptr: int = 0
        self._logger.info("SMReceiver initialized.")

    def start(self) -> None:
        # Acquire shared memory blocks
        self._metadata_block = self.acquire_shm(METADATA_BLOCK_NAME, dts.Metadata.size(), access=mmap.ACCESS_WRITE)
        self._gaze_data_block = self.acquire_shm(GAZE_DATA_BLOCK_NAME, dts.GazeData.size() * GAZE_DATA_BLOCK_CNT, access=mmap.ACCESS_READ)

        if self._metadata_block is None or self._gaze_data_block is None:
            raise RuntimeError("Failed to acquire shared memory blocks.")

        self._logger.info("Metadata (%s) and Gaze Data (%s) blocks acquired.", METADATA_BLOCK_NAME, GAZE_DATA_BLOCK_NAME)

        self._console.start()
        # Start main thread
        if self._thread is None:
            self._running = True
            self._thread = threading.Thread(target=self._run)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._running = False
            self._thread.join()
            self._thread = None
    
    def _run(self) -> None:
        while self._running:
            # Check the stream_ready flag in metadata
            metadata = self.read_metadata_block()
            self._ready = (metadata.stream_ready == 1)

            if not self._ready:
                self._console.print("Waiting for stream to be ready...", use_spinner=True)
                time.sleep(0.1)
                continue

            # Here you can add code to read gaze data if needed
            if metadata.active_data_cnt == 0:
                self._console.print("No new gaze data available...", use_spinner=True)
                time.sleep(0.1)
                continue

            break

        self._monitor.start()

        while self._running:
            metadata = self.read_metadata_block()
            if metadata.active_data_cnt > 0:
                gaze_datas = self.read_gaze_data_blocks(int(metadata.active_data_cnt))
                # Notify listeners
                with self._lock:
                    for listener in self._listeners:
                        listener(gaze_datas)
                # Update monitor
                self._monitor.update(len(gaze_datas))
                # reset cnt
                metadata.active_data_cnt = np.uint8(0)
                self.write_metadata_cnt(metadata)
                self._console.print(f"Gaze Data Rate: {self._monitor.get_data_rate():.1f} Hz"
                                    f" | Avg Data Count: {self._monitor.get_avg_data_cnt():.1f}"
                                    f" | Total: {self._monitor.get_total_packets()}     ", use_spinner=True)
                # self.pretty_print_gaze_data(gaze_datas[-1])  # Print the latest gaze data

        self._monitor.reset()
    
    def acquire_shm(self, block_name: str, block_size: int, access: int = mmap.ACCESS_DEFAULT) -> mmap.mmap:
        """
        Acquire a shared memory block by its name.

        Args:
            block_name (str): The name of the shared memory block.
            block_size (int): The size of the shared memory block.

        Returns:
            mmap.mmap: The acquired shared memory block.
        """
        shm = mmap.mmap(-1, block_size, tagname=block_name, access=access)
        return shm
    
    def read_metadata_block(self) -> dts.Metadata:
        """
        Read the metadata block from shared memory.

        Returns:
            dts.Metadata: An instance of the Metadata dataclass containing the metadata fields and their values.
        """
        assert self._metadata_block is not None, "Metadata block is not initialized."
        self._metadata_block.seek(0)
        data = self._metadata_block.read(dts.Metadata.size())
        return dts.Metadata.from_buffer(data)
    
    def read_gaze_data_blocks(self, count: int = 1) -> list[dts.GazeData]:
        """
        Read gaze data blocks from shared memory.

        Returns:
            list[dts.GazeData]: A list of GazeData dataclass instances containing the gaze data fields and their values.
        """
        gaze_datas: list[dts.GazeData] = []
        block_size = dts.GazeData.size()

        assert self._gaze_data_block is not None, "Gaze data block is not initialized."

        for _ in range(count):
            self._gaze_data_block.seek(self._gaze_data_ptr)
            data = self._gaze_data_block.read(block_size)
            gaze_data = dts.GazeData.from_buffer(data)
            gaze_datas.append(gaze_data)

            self._gaze_data_ptr += block_size
            # Check for circular buffer wrap-around
            if self._gaze_data_ptr >= GAZE_DATA_BLOCK_CNT * block_size:
                self._gaze_data_ptr = 0

        return gaze_datas
    
    def write_metadata_cnt(self, metadata: dts.Metadata) -> None:
        """
        Write the updated active_data_cnt back to the metadata block.
        """
        assert self._metadata_block is not None, "Metadata block is not initialized."
        self._metadata_block.seek(2)  # Offset for active_data_cnt
        self._metadata_block.write(metadata.active_data_cnt.tobytes())
        self._metadata_block.flush()


class ZMQReceiver(PyReceiverBase):
    """
    ZeroMQ Receiver for Gaze Data using pub/sub socket architecture.
    """
    SOCKET_SUB_FILTER = ""

    def __init__(self, address: str = "tcp://localhost:5555"):
        super().__init__()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, self.SOCKET_SUB_FILTER)
        self._socket.connect(address)
        self._logger.info("ZMQReceiver initialized.")

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run)
            self._thread.start()
            self._monitor.start()
            self._logger.info("ZMQReceiver started and connected to %s", self._socket.getsockopt_string(zmq.LAST_ENDPOINT))

    def stop(self) -> None:
        if self._thread is not None:
            self._running = False
            self._thread.join()
            self._thread = None
            self._logger.info("ZMQReceiver stopped.")

    def _run(self) -> None:
        while self._running:
            try:
                message = self._socket.recv_json(flags=zmq.NOBLOCK)
                gaze_data = dts.GazeData(**message)
                # Notify listeners
                with self._lock:
                    for listener in self._listeners:
                        listener([gaze_data])
                # Update monitor
                self._monitor.update(1)
                self._console.print(f"Gaze Data Rate: {self._monitor.get_data_rate():.1f} Hz"
                                    f" | Avg Data Count: {self._monitor.get_avg_data_cnt():.1f}"
                                    f" | Total: {self._monitor.get_total_packets()}     ", use_spinner=True)
                # self.pretty_print_gaze_data(gaze_data)  # Print the latest gaze data
            except zmq.Again:
                time.sleep(0.01)  # No message received, wait a bit
    
class Monitor:
    def __init__(self):
        self._last_timestamp: float = 0.0
        self._data_rate: float = 0.0
        self._data_cnt: int = 0
        self._update_cnt: int = 0
        self._data_cnt_avg: float = 0.0
        self.total_packets: int = 0

    def update(self, packets_received: int):
        if self._last_timestamp == 0:
            self.start()
            return
        self._data_cnt += packets_received
        self._update_cnt += 1
        self.total_packets += packets_received
        if time.time() - self._last_timestamp >= 1.0:
            self._data_rate = self._data_cnt / (time.time() - self._last_timestamp)
            self._data_cnt_avg = self._data_cnt / self._update_cnt if self._update_cnt > 0 else 0.0
            self._data_cnt = 0
            self._update_cnt = 0
            self._last_timestamp = time.time()

    def start(self):
        self.reset()
        self._last_timestamp = time.time()

    def reset(self):
        self._last_timestamp = 0.0
        self._data_rate = 0.0
        self._data_cnt = 0
        self._update_cnt = 0
        self._data_cnt_avg = 0.0

    def get_data_rate(self) -> float:
        return self._data_rate

    def get_avg_data_cnt(self) -> float:
        return self._data_cnt_avg

    def get_total_packets(self) -> int:
        return self.total_packets