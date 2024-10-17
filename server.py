import grpc
from concurrent import futures
import object_detection_pb2_grpc, object_detection_pb2
from detect import run_model
import warnings
import psutil
import time

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

def get_energy_efficiency():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    return f"CPU Usage: {cpu_percent}%, Memory Usage: {memory_info.percent}%"


class ObjectDetectionServiceServicer(object_detection_pb2_grpc.ObjectDetectionServiceServicer):
    def DetectObjects(self, request, context):
        try:
            print(f"Received request: {request}")
            model_type = request.model_type
            detected_objects, latency, accuracy = run_model(request.image_path, model_type)
            energy_efficiency = get_energy_efficiency()
        
            response = object_detection_pb2.DetectionResponse()
            response.accuracy = accuracy
            response.energy_efficiency = energy_efficiency

            # print(f"Detected objects: {detected_objects}")

            for obj in detected_objects:
                detected_object = object_detection_pb2.DetectedObject(
                    label=obj['label'],
                    confidence=obj['confidence'],
                    x=int(obj['x']),  # Ensure it's an int
                    y=int(obj['y']),  # Ensure it's an int
                    width=int(obj['width']),  # Ensure it's an int
                    height=int(obj['height'])  # Ensure it's an int
                )
                response.objects.append(detected_object)
            return response
        
        except Exception as e:
            print(f"Error during object detection: {e}")
            context.set_details(f"Exception calling application: {str(e)}")
            context.set_code(grpc.StatusCode.UNKNOWN)
            return object_detection_pb2.DetectionResponse()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    object_detection_pb2_grpc.add_ObjectDetectionServiceServicer_to_server(
        ObjectDetectionServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server running on port 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
