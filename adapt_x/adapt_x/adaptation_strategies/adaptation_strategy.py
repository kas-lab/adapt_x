from abc import ABC, abstractmethod


class AdaptationStrategy(ABC):
    def __init__(self, strategy_name):
        self.name = strategy_name + "_strategy"
        self.configuration_dict = {}

    def get_name(self):
        return self.name

    @abstractmethod
    def get_monitor_service(self) -> tuple[object, str]:
        pass

    @abstractmethod
    def get_monitor_request(self):
        pass

    @abstractmethod
    def get_initialization_service(self) -> tuple[object, str]:
        pass

    @abstractmethod
    def get_initialization_request(self):
        pass

    @abstractmethod
    def on_initialization(self, result):
        pass

    @abstractmethod
    def on_monitored(self):
        pass

    @abstractmethod
    def on_requirements(self):
        pass

    @abstractmethod
    def on_task_context(self):
        pass

    @abstractmethod
    def on_task_complete(self):
        pass

    @abstractmethod
    def on_receive_task(self):
        pass
