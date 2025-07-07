import json

class DominionConfiguration:
    """
    Handles the Dominion configuration protocol.
    """

    class Request:
        def __init__(self, user_name, peer_id):
            self.user_name = user_name
            self.peer_id = peer_id

        def to_json(self):
            return json.dumps(self.__dict__)

    class Response:
        def __init__(self, data):
            self.__dict__ = json.loads(data)
