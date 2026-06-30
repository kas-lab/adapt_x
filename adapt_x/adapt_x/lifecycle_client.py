from rclpy.client import SrvTypeRequest
from rclpy.client import SrvTypeResponse
from rclpy.client import Client
from rclpy.task import Future
from rclpy.lifecycle.managed_entity import ManagedEntity
from rclpy.lifecycle import TransitionCallbackReturn
from functools import wraps

class BetterManagedEntity(ManagedEntity):
    """A simple managed entity that only sets a flag when activated/deactivated."""

    def __init__(self):
        self._enabled = False

    def on_activate(self, state) -> TransitionCallbackReturn:
        self._enabled = True
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state) -> TransitionCallbackReturn:
        self._enabled = False
        return TransitionCallbackReturn.SUCCESS

    @property
    def is_activated(self):
        return self._enabled

    @staticmethod
    def when_enabled(wrapped=None, *, when_not_enabled=None):
        def decorator(wrapped):
            @wraps(wrapped)
            def only_when_enabled_wrapper(self: BetterManagedEntity, *args, **kwargs):
                if not self._enabled:
                    if when_not_enabled is not None:
                        return when_not_enabled()
                    return
                return wrapped(self, *args, **kwargs)
            return only_when_enabled_wrapper
        if wrapped is None:
            return decorator
        return decorator(wrapped)


class LifecycleClient(BetterManagedEntity, Client):
    """Managed publisher entity."""

    def __init__(self, *args, **kwargs):
        BetterManagedEntity.__init__(self)
        Client.__init__(self, *args, **kwargs)

    def wait_for_service(self, timeout_sec: float = None) -> bool:
        print(timeout_sec)
        result = Client.wait_for_service(self, timeout_sec=timeout_sec)
        print(f"wait_for_service returned: {result}")
        return result

    @BetterManagedEntity.when_enabled
    def call_async(self, request: SrvTypeRequest) -> Future:
        """
        Call a service asynchronously if the lifecycle client is enabled.
        """
        x = Client.call_async(self, request)
        return x

    @BetterManagedEntity.when_enabled
    def call(self, request: SrvTypeRequest) -> SrvTypeResponse:
        """
        """
        return Client.call(self, request)
