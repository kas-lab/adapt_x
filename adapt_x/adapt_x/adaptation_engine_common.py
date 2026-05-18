import rclpy
from std_srvs.srv import Trigger
from statistics import mean


def client_request_init(node, client, request, identifier, timeout_sec=10):
    if (client.wait_for_service(timeout_sec=15.0)):
        node.get_logger().info(f"Sending {identifier} query to Typedb...")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
        if future.result() is not None:
            node.get_logger().info(f"Received response for {identifier} query.")
            result = future.result()

            if hasattr(result, 'success') and not result.success:
                node.get_logger().error(f"Typedb {identifier} query failed.")
                raise RuntimeError(f"Typedb {identifier} query had success false.")
            return result
        else:
            node.get_logger().error(f"Service call failed for {identifier} query. Potential timeout")
            raise RuntimeError(f"Typedb query service call for {identifier} failed.")
    else:
        node.get_logger().error(f"Typedb query service not available for {identifier} query.")
        raise RuntimeError(f"Typedb query service not available for {identifier}")


def request_measures(node):
    create_measures_client = node.create_client(Trigger, f"/{node.kb_node}/create_measures")
    if (create_measures_client.wait_for_service()):
        # java.lang.System.out.println("createMeasures Service is available");
        future = create_measures_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future)
        response = future.result()

        if response.success is False:
            raise RuntimeError("Create Measures Failed")

        node.destroy_client(create_measures_client)
        return response
    else:
        raise RuntimeError("Service is not available")


def mean_fulfillment(reqs_msg, system_name):
    return mean(float(req.fulfillment) for req in reqs_msg.requirements
                if req.system_name == system_name)
