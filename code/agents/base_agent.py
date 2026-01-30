#base_agent.py

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, player_id, role):
        self.player_id = player_id
        self.role = role

    @abstractmethod
    def get_action(self, observation, available_actions):
        """
        現在の観測情報と可能な行動リストから行動を決定する
        """
        pass