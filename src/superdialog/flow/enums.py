from enum import Enum


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class ActionTriggerType(str, Enum):
    ON_ENTER = "on_enter"
    ON_EXIT = "on_exit"
