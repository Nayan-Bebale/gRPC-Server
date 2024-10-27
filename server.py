import grpc
from concurrent import futures
from server_data import display_ip_info  # Importing server_data dictionary
import object_detection_pb2_grpc, object_detection_pb2
from detect import run_model
import warnings
import psutil
import logging
import os

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_energy_efficiency():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    return cpu_percent, memory_info.percent

class ObjectDetectionServiceServicer(object_detection_pb2_grpc.ObjectDetectionServiceServicer):
    def DetectObjects(self, request, context):
        try:
            # logging.info(f"Received request: {request}")
            # logging.info(f"Image Path: {request.image_path}")
            # logging.info(f"Model Type: {request.model_type}")
            model_type = request.model_type
            detected_objects, latency, accuracy, throughput, energy_required, power_watts = run_model(request.image_path, model_type)
            # print(detected_objects, latency, accuracy, throughput, energy_required, power_watts)
            cpu_usage, memory_usage = get_energy_efficiency()

            # Initialize the response object
            response = object_detection_pb2.DetectionResponse()
            response.accuracy = accuracy
            response.cpu_usage = cpu_usage
            response.memory_usage = memory_usage
            response.latency = latency
            response.throughput = throughput
            response.energy_required = energy_required
            response.power_watts = power_watts
            # Adding detected objects
            for obj in detected_objects:
                detected_object = object_detection_pb2.DetectedObject(
                    label=obj['label'],
                    confidence=obj['confidence'],
                    x=int(obj['x']),
                    y=int(obj['y']),
                    width=int(obj['width']),
                    height=int(obj['height'])
                )
                response.objects.append(detected_object)
            all_data = display_ip_info()
            # print(all_data)
            # Adding server_data (display_ip_info)
            # Populate the server_data map in the response
            for key, value in all_data.items():
                response.server_data[key] = str(value)  # Ensure all values are strings
            return response
        
        except Exception as e:
            logging.error(f"Error during object detection: {e}")
            context.set_details(f"Exception calling application: {str(e)}")
            context.set_code(grpc.StatusCode.UNKNOWN)
            return object_detection_pb2.DetectionResponse()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    object_detection_pb2_grpc.add_ObjectDetectionServiceServicer_to_server(
        ObjectDetectionServiceServicer(), server)
    
    try:
        server.add_insecure_port('0.0.0.0:50505')
        server.start()
        logging.info("Server running on port 50505...")
        server.wait_for_termination()
    except Exception as e:
        logging.error(f"Failed to start server: {e}")

if __name__ == '__main__':
    serve()
