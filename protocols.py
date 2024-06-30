from components import Network, Node, Channel
from typing import List, Tuple, Dict, Optional
import numpy as np
import random

class Protocol:
    def execute(self, nodes: List[Node], channels: List[Channel]) -> int:
        raise NotImplementedError("Must be implemented by subclass")

class RandomAccessProtocol(Protocol):
    def execute(self, nodes: List[Node], channels: List[Channel]) -> int:
        success = 0
        for node in nodes:
            if channels:  # Ensure there are channels available
                channel = random.choice(channels)
                success += node.send_data(channel.gains[node])
        return success

class OneToOneStableMatching(Protocol):
    def __init__(self, mode: str = 'node') -> None:
        super().__init__()
        self.mode = mode

    def stable_matching(self, nodes: List[Node], channels: List[Channel]) -> List[Tuple[Node, Channel]]:
        sending_nodes = nodes
        node_preferences = {node: sorted(channels, key=lambda channel: channel.gains[node]) for node in sending_nodes}
        channel_preferences = {channel: sorted(sending_nodes, key=lambda node: -node.energy) for channel in channels}

        if self.mode == 'node':
            proposers = sending_nodes
            proposer_preferences = node_preferences
            receivers = channels
            receiver_preferences = channel_preferences
        elif self.mode == 'channel':
            proposers = channels
            proposer_preferences = channel_preferences
            receivers = sending_nodes
            receiver_preferences = node_preferences

        receiver_partner = {receiver: None for receiver in receivers}
        proposer_free = {proposer: True for proposer in proposers}
        free_proposers = len(proposers)

        while free_proposers > max(0, len(proposers) - len(receivers)):
            for proposer in proposers:
                if not proposer_free[proposer]:
                    continue

                for receiver in proposer_preferences[proposer]:
                    if receiver_partner[receiver] is None:
                        receiver_partner[receiver] = proposer
                        proposer_free[proposer] = False
                        free_proposers -= 1
                        break
                    else:
                        current_partner = receiver_partner[receiver]
                        if receiver_preferences[receiver].index(proposer) < receiver_preferences[receiver].index(current_partner):
                            receiver_partner[receiver] = proposer
                            proposer_free[proposer] = False
                            proposer_free[current_partner] = True
                            break
                        
        if self.mode == 'node':
            return [(proposer, receiver) for receiver, proposer in receiver_partner.items() if proposer is not None]      
        return [(receiver, proposer) for receiver, proposer in receiver_partner.items() if proposer is not None]
    
    def execute(self, nodes: List[Node], channels: List[Channel]) -> int:
        matching = self.stable_matching(nodes, channels)
        success = 0
        for node, channel in matching:
            if node is None:
                continue
            success += node.send_data(channel.gains[node])
        return success
    
class OneToManyStableMatching(Protocol):
    def __init__(self, max_matches_per_receiver=3):
        super().__init__()
        self.max_matches_per_receiver = max_matches_per_receiver

    def stable_matching(self, nodes: List[Node], channels: List[Channel]) -> List[Tuple[Node, Channel]]:
        sending_nodes = nodes
        proposers = sending_nodes
        receivers = channels

        proposer_preferences = {node: sorted(channels, key=lambda channel: channel.gains[node]) for node in sending_nodes}
        receiver_preferences = {channel: sorted(sending_nodes, key=lambda node: -node.energy) for channel in channels}

        receiver_partners = {receiver: [] for receiver in receivers}
        proposer_free = {proposer: True for proposer in proposers}
        free_proposers = len(proposers)

        while free_proposers > 0:
            for proposer in proposers:
                if not proposer_free[proposer]:
                    continue

                for receiver in proposer_preferences[proposer]:
                    if len(receiver_partners[receiver]) < self.max_matches_per_receiver:
                        receiver_partners[receiver].append(proposer)
                        proposer_free[proposer] = False
                        free_proposers -= 1
                        break
                    else:
                        worst_partner = max(receiver_partners[receiver], key=lambda partner: receiver_preferences[receiver].index(partner))
                        if receiver_preferences[receiver].index(proposer) < receiver_preferences[receiver].index(worst_partner):
                            receiver_partners[receiver].remove(worst_partner)
                            receiver_partners[receiver].append(proposer)
                            proposer_free[proposer] = False
                            proposer_free[worst_partner] = True
                            break

        return [(proposer, receiver) for receiver, partners in receiver_partners.items() for proposer in partners if proposer is not None]

    def execute(self, nodes: List[Node], channels: List[Channel]) -> int:
        matching = self.stable_matching(nodes, channels)
        success = 0
        for node, channel in matching:
            if node is None:
                continue
            success += node.send_data(channel.gains[node])
        return success

class OptimalSellingMechanism(Protocol):
    def __init__(self, mode: str = "energy") -> None:
        super().__init__()
        self.mode = mode
        
        
    def value(self, node: Node, channels: List[Channel]) -> float:
        if self.mode == "energy":
            return node.energy
        elif self.mode == "probability":
            best_channel = min(channels, key=lambda ch: ch.gains[node])
            return node.probability_of_success(best_channel.gains[node])
        else:
            return 0
    
    def cdf(self, value: float, node: Node, num_channels: int) -> float:
        n = num_channels
        if self.mode == "energy":
            return node.dist.cdf(value)
        elif self.mode == "probability":
            return (1 - value ** (n - 1)) * n / (n - 1)
        else:
            return 1
        
    def pdf(self, value: float, node: Node, num_channels: int) -> float:
        n = num_channels
        if self.mode == "energy":
            return node.dist.pdf(value)
        elif self.mode == "probability":
            return (n * value - value ** n) / (n - 1)
        else:
            return 1
    
    
    def c(self, node: Node, channels: List[Channel]) -> float:
        v = self.value(node, channels)
        return v - (1 - self.cdf(v, node, len(channels))) / self.pdf(v, node, len(channels))
    
    # In effect, the one with the highest value will be the winner
    def q(self, Nodes: List[Node], channels: List[Channel]) -> Optional[int]:
        C = [self.c(node, channels) for node in Nodes]
        if max(C) < 0:
            return None
        winners = [np.argmax(C)]
        return random.choice(winners)
    
    def matching(self, nodes: List[Node], channels: List[Channel]) -> List[Tuple[Node, Channel]]:
        sending_nodes = nodes
        waiting_nodes = sending_nodes[:]
        free_channels = channels[:]
        matching = []

        while len(waiting_nodes) > 0:
            winner_index = self.q(waiting_nodes, free_channels)
            if winner_index is None:
                break
            winner = waiting_nodes.pop(winner_index)
            best_channel = min(free_channels, key=lambda ch: ch.gains[winner])
            matching.append((winner, best_channel))

        return matching
    
    def execute(self, nodes: List[Node], channels: List[Channel]) -> int:
        matching = self.matching(nodes, channels)
        success = 0
        for node, channel in matching:
            if node is None:
                continue
            success += node.send_data(channel.gains[node])
        return success

