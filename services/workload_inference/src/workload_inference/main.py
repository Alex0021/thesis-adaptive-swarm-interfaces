import argparse
import contextlib
import logging
import os
import time

# Import torch before PyQt6 to avoid DLL conflicts on Windows
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

os.environ["QT_API"] = "PyQt6"  # Ensure PyQt6 is used for matplotlib backend

import debugpy
import numpy as np
import zmq
from PyQt6.QtWidgets import QApplication

import workload_inference.experiments.data_structures as dts
from workload_inference.generator import FakeGazeGenerator
from workload_inference.processing import DataProcessor
from workload_inference.py_receiver import (
    SMReceiver,
    SMReceiverCircularBuffer,
    ZMQReceiver,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s::%(message)s",
        handlers=[logging.StreamHandler()],
    )


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run workload inference experiment (N-back or gate racing)"
    )
    parser.add_argument(
        "--experiment",
        choices=["nback", "gates"],
        default="nback",
        help="Which experiment to run (default: nback)",
    )
    args = parser.parse_args()

    app = QApplication([])
    setup_logging()
    logger = logging.getLogger()
    logger.info("Workload Inference Service Started (experiment: %s)", args.experiment)

    try:
        # Import the appropriate experiment module
        if args.experiment == "nback":
            from workload_inference.experiments import (
                NBackExperimentManager as ExperimentManager,
            )
            from workload_inference.experiments import (
                NBackExperimentManagerWindow as ExperimentManagerWindow,
            )
        else:  # gate_racing
            from workload_inference.experiments import (
                GateRacingExperimentManager as ExperimentManager,
            )
            from workload_inference.experiments import (
                GateRacingExperimentManagerWindow as ExperimentManagerWindow,
            )

        experiment_manager = ExperimentManager()
        experiment_window = ExperimentManagerWindow(experiment_manager)
        experiment_window.show()
        # fake_data_generator = FakeGazeGenerator(
        #     callback=experiment_window._gaze_visualizer.datas_callback,
        #     frequency=60.0,
        #     noise=0.05,
        #     speed=2.5,
        #     pupil_mean=3.5,
        # )
        # fake_data_generator.start()
        # experiment_window.attach_listeners()
        experiment_window.start()

        # receiver = ZMQReceiver()
        # def print_drone_data(datas: list[dts.DroneData]):
        #     for d in datas:
        #         pos = np.array([d.position_x, d.position_y, d.position_z])
        #         print(pos)

        # receiver = SMReceiver(
        #     mmap_name=dts.DRONE_DATA_BLOCK_NAME,
        #     datatype=dts.DroneData,
        #     update_rate=2,
        #     listeners=[print_drone_data],
        #     block_count=dts.DRONE_COUNT,
        # )
        # receiver = SMReceiverCircularBuffer(
        #     data_mmap_name="TobiiUnityGazeData",
        #     mmap_name="TobiiUnityMetadata",
        #     datatype=dts.GazeData,
        #     update_rate=60,
        #     block_count=100,
        #     with_timestamps=True,
        # )
        # receiver.start()
    # receiver.register_listener(experiment_manager.datas_callback)
    # # receiver.register_listener(data_processor.datas_callback)
    # receiver.register_listener(visualizer.canvas.datas_callback)
    # receiver.start()
    # experiment_manager.start_recording()
    except Exception as e:
        logger.exception("%s", e, stack_info=True)
        return
    try:
        # while True:
        #     print(f"Data Processor Samples: {data_processor.get_num_samples()}")
        #     try:
        #         print(data_processor.get_samples()[-1]) # Print the latest sample
        #     except IndexError as e:
        #         pass
        #     time.sleep(1)
        # visualizer.showMaximized()
        app.exec()
    except KeyboardInterrupt:
        pass
    finally:
        # receiver.stop()
        logger.info("Workload Inference Service Stopped")

    # if os.environ.get("PYTHONDEBUG", "0") == "1":
    #     print("Waiting for debugger to attach...")
    #     debugpy.listen(("0.0.0.0", 5678))
    #     debugpy.wait_for_client()

    # context = zmq.Context()
    # socket = context.socket(zmq.SUB)
    # if os.environ.get("IS_DOCKER", "0") == "1":
    #     socket.connect("tcp://host.docker.internal:5555")
    # else:
    #     socket.connect("tcp://localhost:5555")
    # socket.setsockopt_string(zmq.SUBSCRIBE, "")

    # while True:
    #     data = socket.recv_json()
    #     print(f"Received data: {data}\n")
    #     time.sleep(0.01)


if __name__ == "__main__":
    main()
