import struct
import threading
import mmap
import data_structures as dts
import time

METADATA_BLOCK_NAME = "UnityToPython_Metadata"
GAZE_DATA_BLOCK_NAME = "UnityToPython_GazeData"
GAZE_DATA_BLOCK_CNT = 100

class PyReceiver:

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._running: bool = False
        self._ready: bool = False
        self._metadata_block: mmap.mmap | None = None
        self._gaze_data_block: mmap.mmap | None = None
        self._gaze_data_ptr: int = 0

    def start(self) -> None:
        # Acquire shared memory blocks
        self._metadata_block = self.acquire_shm(METADATA_BLOCK_NAME, sum(dts.sm_metadata.values()))
        self._gaze_data_block = self.acquire_shm(GAZE_DATA_BLOCK_NAME, sum(dts.sm_gaze_data.values()), access=mmap.ACCESS_READ)

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
            self._ready = (metadata["stream_ready"] == 1)

            if not self._ready:
                print("\rWaiting for Unity stream...", end="")
                continue

            # Here you can add code to read gaze data if needed
            if metadata["active_data_cnt"] == 0:
                print("\rWaiting for gaze data...", end="")
                continue
            
            gaze_datas = self.read_gaze_data_blocks(metadata["active_data_cnt"])
            # reset cnt
            metadata["active_data_cnt"] = 0
            self.write_metadata_cnt(metadata)
            print(f"\r{gaze_datas[0]}", end="")
    
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
    
    def read_metadata_block(self) -> dict[str, int]:
        """
        Read the metadata block from shared memory.

        Returns:
            dict[str, int]: A dictionary containing the metadata fields and their values.
        """
        metadata: dict[str, int] = {}
        self._metadata_block.seek(0)
        data = struct.unpack('BBB', self._metadata_block.read(sum(dts.sm_metadata.values())))
        metadata["stream_ready"] = data[0]
        metadata["calibration_ok"] = data[1]
        metadata["active_data_cnt"] = data[2]
        return metadata
    
    def read_gaze_data_blocks(self, count: int = 1) -> list[dict[str, float]]:
        """
        Read gaze data blocks from shared memory.

        Returns:
            list[dict[str, float]]: A list of dictionaries containing the gaze data fields and their values.
        """
        gaze_datas: list[dict[str, float]] = []
        block_size = sum(dts.sm_gaze_data.values())
        self._gaze_data_block.seek(self._gaze_data_ptr)

        for _ in range(count):
            data = struct.unpack('dffffffBBff', self._gaze_data_block.read(block_size))
            gaze_data: dict[str, float] = {}
            gaze_data["timestamp"] = data[0]
            gaze_data["left_gaze_point"] = (data[1], data[2], data[3])
            gaze_data["right_gaze_point"] = (data[4], data[5], data[6])
            gaze_data["left_point_screen"] = (data[7], data[8])
            gaze_data["right_point_screen"] = (data[9], data[10])
            gaze_data["left_validity"] = data[11]
            gaze_data["right_validity"] = data[12]
            gaze_data["left_pupil_diameter"] = data[13]
            gaze_data["right_pupil_diameter"] = data[14]
            gaze_datas.append(gaze_data)

            self._gaze_data_ptr += block_size
            # Check for circular buffer wrap-around
            if self._gaze_data_ptr >= GAZE_DATA_BLOCK_CNT * block_size:
                self._gaze_data_ptr = 0

        self._gaze_data_ptr = (self._gaze_data_ptr + 1) % GAZE_DATA_BLOCK_CNT
        return gaze_datas
    
    def write_metadata_cnt(self, metadata: dict[str, int]) -> None:
        """
        Write the updated active_data_cnt back to the metadata block.
        """
        if not "active_data_cnt" in metadata:
            return
        self._metadata_block.seek(2)  # Offset for active_data_cnt
        self._metadata_block.write(struct.pack('B', metadata["active_data_cnt"]))
        self._metadata_block.flush()
    

